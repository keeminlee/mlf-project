"""
gnn_ik.py

GNN-based inverse kinematics (IK) model for the KUKA iiwa arm, using
trajectory-style data from kuka_fk_dataset.py --data-type traj.

Dataset (TRAJ mode CSV):
    Columns:
      pose (xyz or xyz+quat),
      q_prev_0..6,
      q_curr_0..6

Model:
    - Represent the robot as a small graph:
        * 1 pose node
        * 7 joint nodes (one per joint)
        * edges for serial chain (joint_i <-> joint_{i+1})
        * edges from pose node to each joint

    - Node features:
        pose node:
            [pose, 0, 0]
        joint node j:
            [0...0 (pose_dim zeros), q_prev_j, j/6]

      where pose_dim is 3 (xyz) or 7 (xyz+quat).

    - GNN message passing over this graph.
    - Global graph embedding via mean pooling.
    - MLP head outputs Δq (7D vector).

Objective:
    Given (pose_t, q_prev), predict Δq such that:
        q_pred = q_prev + Δq ≈ q_curr
    Loss:
        ik_loss      = MSE(q_pred, q_curr)
        move_loss    = MSE(Δq, 0)    # encourages small joint changes
        total loss   = ik_loss + lambda_movement * move_loss

This script provides:
- KukaTrajGraphDataset:  per-sample graph construction
- KukaTrajGNNDataModule: LightningDataModule for GNN IK
- IK_GNN:                LightningModule GNN model
- run_gnn_grid_search:   optional grid search using IKGridSearch
- main():                simple single-run training CLI
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import argparse
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

from grid_search import IKGridSearch


# ------------------------------------------------------------------------------
# Data loading utilities (TRAJ mode only)
# ------------------------------------------------------------------------------

from data_utils import load_traj_csv


# ------------------------------------------------------------------------------
# Graph dataset for PyTorch Geometric
# ------------------------------------------------------------------------------

class KukaTrajGraphDataset(Dataset):
    """
    PyTorch Geometric Dataset for trajectory IK data.

    Each sample is converted into a small graph with:
        - 1 pose node
        - 7 joint nodes

    Node features:
        pose node (index 0):
            [pose, 0, 0]
        joint node j in {0..6} (graph node index j+1):
            [0...0 (pose_dim zeros), q_prev_j, j/6]

    Edges:
        - chain edges between joint nodes:
            (1 <-> 2), (2 <-> 3), ..., (6 <-> 7)
        - pose node connected to every joint:
            (0 <-> j) for j = 1..7

    Stored in `Data` as:
        x:        (8, pose_dim+2) node features
        edge_index: (2, E) graph edges
        y:        (7,) target q_curr
        q_prev:   (7,) previous joints
    """

    def __init__(
        self,
        poses: np.ndarray,
        q_prev: np.ndarray,
        q_curr: np.ndarray,
        pose_dim: int,
    ):
        assert poses.shape[0] == q_prev.shape[0] == q_curr.shape[0]
        self.poses = torch.from_numpy(poses)       # (N, pose_dim)
        self.q_prev = torch.from_numpy(q_prev)     # (N, 7)
        self.q_curr = torch.from_numpy(q_curr)     # (N, 7)
        self.pose_dim = pose_dim
        self.n_joints = 7
        self.num_nodes = 1 + self.n_joints        # 1 pose + 7 joints
        self.node_feat_dim = pose_dim + 2         # pose components + q_prev + joint index scalar

        # Pre-compute a shared edge_index for all graphs
        self.edge_index = self._build_edge_index()

    def _build_edge_index(self) -> torch.Tensor:
        edges = []

        # Chain edges between joints: (1 <-> 2), ..., (6 <-> 7)
        for j in range(1, self.n_joints):  # joints are nodes 1..7
            a = j
            b = j + 1
            edges.append((a, b))
            edges.append((b, a))

        # Pose node (0) connected to every joint node (1..7)
        for j in range(1, self.n_joints + 1):
            edges.append((0, j))
            edges.append((j, 0))

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return edge_index

    def __len__(self) -> int:
        return self.poses.shape[0]

    def __getitem__(self, idx: int) -> Data:
        pose = self.poses[idx]      # (pose_dim,)
        q_prev = self.q_prev[idx]   # (7,)
        q_curr = self.q_curr[idx]   # (7,)

        # Build node features x: (8, node_feat_dim)
        x = torch.zeros((self.num_nodes, self.node_feat_dim), dtype=torch.float32)

        # Pose node features: [pose, 0, 0]
        x[0, :self.pose_dim] = pose

        # Joint nodes features:
        #   [0...0, q_prev_j, j/6]  for joint j
        for j in range(self.n_joints):
            node_idx = j + 1
            x[node_idx, self.pose_dim] = q_prev[j]
            x[node_idx, self.pose_dim + 1] = float(j) / float(self.n_joints - 1)  # j / 6

        data = Data(
            x=x,
            edge_index=self.edge_index,
        )
        data.y = q_curr
        data.q_prev = q_prev
        return data


# ------------------------------------------------------------------------------
# Lightning DataModule for GNN IK (TRAJ only)
# ------------------------------------------------------------------------------

class KukaTrajGNNDataModule(pl.LightningDataModule):
    """
    LightningDataModule for trajectory-based GNN IK training.

    This assumes the CSV was generated with:
        kuka_fk_dataset.py --data-type traj ...
    """

    def __init__(
        self,
        csv_path: str | Path,
        batch_size: int = 128,
        use_orientation: bool = False,
        val_frac: float = 0.15,
        test_frac: float = 0.15,
        seed: int = 42,
    ):
        super().__init__()
        self.csv_path = str(csv_path)
        self.batch_size = batch_size
        self.use_orientation = use_orientation
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.seed = seed

        self.pose_dim: Optional[int] = None
        self.n_joints: int = 7

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        poses, q_prev, q_curr, pose_dim = load_traj_csv(
            self.csv_path,
            use_orientation=self.use_orientation,
        )
        self.pose_dim = pose_dim

        N = poses.shape[0]
        rng = np.random.default_rng(self.seed)
        indices = np.arange(N)
        rng.shuffle(indices)

        n_val = int(self.val_frac * N)
        n_test = int(self.test_frac * N)
        n_train = N - n_val - n_test

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        train_dataset = KukaTrajGraphDataset(
            poses[train_idx], q_prev[train_idx], q_curr[train_idx], pose_dim
        )
        val_dataset = KukaTrajGraphDataset(
            poses[val_idx], q_prev[val_idx], q_curr[val_idx], pose_dim
        )
        test_dataset = KukaTrajGraphDataset(
            poses[test_idx], q_prev[test_idx], q_curr[test_idx], pose_dim
        )

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def train_dataloader(self):
        return GeoDataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return GeoDataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def test_dataloader(self):
        return GeoDataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )


# ------------------------------------------------------------------------------
# GNN IK model (LightningModule)
# ------------------------------------------------------------------------------

class IK_GNN(pl.LightningModule):
    """
    GNN-based IK model with Δq regression and movement penalty.

    Forward:
        data -> (Δq_pred), shape (B, 7)
        q_pred = q_prev + Δq_pred

    Loss:
        ik_loss    = MSE(q_pred, q_curr)
        move_loss  = MSE(Δq_pred, 0)
        total      = ik_loss + lambda_movement * move_loss
    """

    def __init__(
        self,
        node_input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        output_dim: int = 7,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        lambda_movement: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.node_input_dim = node_input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lambda_movement = lambda_movement

        # Input projection for node features
        self.lin_in = nn.Linear(node_input_dim, hidden_dim)

        # GNN layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # Head to map graph embedding -> Δq (7D)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, data: Data) -> torch.Tensor:
        """
        Args:
            data: a PyG Data or Batch object with attributes:
                  x: node features, shape (num_nodes_total, node_input_dim)
                  edge_index: (2, E)
                  batch: graph index per node, shape (num_nodes_total,)

        Returns:
            dq_pred: (batch_size, output_dim) predicted Δq
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.lin_in(x)
        x = F.relu(x)

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Graph-level embedding via mean pooling
        g_emb = global_mean_pool(x, batch)  # (B, hidden_dim)

        dq_pred = self.head(g_emb)          # (B, output_dim)
        return dq_pred

    def _compute_losses(self, data: Data, dq_pred: torch.Tensor):
        # data.q_prev and data.y are concatenated over the batch dimension by PyG,
        # so they come in as shape (batch_size * 7,). We reshape to (batch_size, 7).
        q_prev = data.q_prev.view(-1, self.output_dim)  # (B, 7)
        q_curr = data.y.view(-1, self.output_dim)       # (B, 7)

        q_pred = q_prev + dq_pred   # predicted joint angles (B, 7)

        ik_loss = F.mse_loss(q_pred, q_curr)
        move_loss = F.mse_loss(dq_pred, torch.zeros_like(dq_pred))

        loss = ik_loss + self.lambda_movement * move_loss
        return ik_loss, move_loss, loss

    def training_step(self, batch: Data, batch_idx: int):
        dq_pred = self(batch)
        ik_loss, move_loss, loss = self._compute_losses(batch, dq_pred)

        self.log("train_ik_loss", ik_loss, on_epoch=True, prog_bar=False)
        self.log("train_move_loss", move_loss, on_epoch=True, prog_bar=False)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Data, batch_idx: int):
        dq_pred = self(batch)
        ik_loss, move_loss, loss = self._compute_losses(batch, dq_pred)

        self.log("val_ik_loss", ik_loss, on_epoch=True, prog_bar=False)
        self.log("val_move_loss", move_loss, on_epoch=True, prog_bar=False)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch: Data, batch_idx: int):
        dq_pred = self(batch)
        ik_loss, move_loss, loss = self._compute_losses(batch, dq_pred)

        self.log("test_ik_loss", ik_loss, prog_bar=False)
        self.log("test_move_loss", move_loss, prog_bar=False)
        self.log("test_loss", loss, prog_bar=False)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=10,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


# ------------------------------------------------------------------------------
# Grid search glue (optional, uses IKGridSearch)
# ------------------------------------------------------------------------------

def build_gnn_datamodule(params: Dict[str, Any], splits: Dict[str, Any]) -> pl.LightningDataModule:
    """
    Factory for IKGridSearch datamodule_builder.

    Args:
        params: contains e.g. batch_size, use_orientation
        splits: contains 'csv_path', 'val_frac', 'test_frac', 'seed'

    Returns:
        KukaTrajGNNDataModule
    """
    csv_path = splits["csv_path"]
    use_orientation = params.get("use_orientation", splits.get("use_orientation", False))

    return KukaTrajGNNDataModule(
        csv_path=csv_path,
        batch_size=params.get("batch_size", 128),
        use_orientation=use_orientation,
        val_frac=splits.get("val_frac", 0.15),
        test_frac=splits.get("test_frac", 0.15),
        seed=splits.get("seed", 42),
    )


def run_gnn_grid_search(
    splits: Dict[str, Any],
    output_dir: str | Path,
    max_epochs: int = 100,
    n_joints: int = 7,
) -> Dict[str, Any]:
    """
    Run grid search over GNN hyperparameters using IKGridSearch.

    Args:
        splits: dict with dataset info:
                  {
                    "csv_path": "...",
                    "val_frac": 0.15,
                    "test_frac": 0.15,
                    "seed": 42,
                    "use_orientation": True/False
                  }
        output_dir: where to save grid-search artifacts
        max_epochs: max epochs per config
        n_joints:  number of joints (7)

    Returns:
        best_result dict with fields: { "params", "best_score", "best_path" }
    """
    param_grid: Dict[str, List[Any]] = {
        "hidden_dim": [64, 128],
        "num_layers": [2, 3],
        "dropout": [0.0, 0.1],
        "learning_rate": [1e-3, 5e-4],
        "weight_decay": [1e-4, 1e-5],
        "lambda_movement": [0.1, 0.2],
        "batch_size": [64, 128],
        "use_orientation": [False, True],
    }

    def model_builder(params: Dict[str, Any], n_joints_inner: int) -> pl.LightningModule:
        # Use datamodule to infer pose_dim -> node_input_dim
        dm = build_gnn_datamodule(params, splits)
        dm.setup()
        pose_dim = dm.pose_dim
        assert pose_dim is not None
        node_input_dim = pose_dim + 2  # pose_dim + q_prev + joint index scalar

        return IK_GNN(
            node_input_dim=node_input_dim,
            hidden_dim=params["hidden_dim"],
            num_layers=params["num_layers"],
            output_dim=n_joints_inner,
            dropout=params.get("dropout", 0.1),
            learning_rate=params.get("learning_rate", 1e-3),
            weight_decay=params.get("weight_decay", 1e-4),
            lambda_movement=params.get("lambda_movement", 0.1),
        )

    runner = IKGridSearch(
        output_dir=output_dir,
        param_grid=param_grid,
        model_builder=model_builder,
        datamodule_builder=build_gnn_datamodule,
        monitor_metric="val_loss",
        mode="min",
        patience=15,
        max_epochs=max_epochs,
        accelerator="auto",
        n_joints=n_joints,
    )

    results = runner.run(splits)
    if not results:
        raise RuntimeError("No successful GNN IK runs in grid search.")
    best = min(results, key=lambda r: r["best_score"])
    return best


# ------------------------------------------------------------------------------
# Single-run CLI
# ------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Train a GNN IK model on trajectory KUKA dataset."
    )
    ap.add_argument(
        "--csv-path",
        type=str,
        default="../data/kuka_traj_dataset.csv",
        help="Path to TRAJ CSV (from kuka_fk_dataset.py --data-type traj).",
    )
    ap.add_argument(
        "--use-orientation",
        action="store_true",
        help="If set, expect xyz+quat (7D pose) instead of xyz (3D).",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for training.",
    )
    ap.add_argument(
        "--max-epochs",
        type=int,
        default=100,
        help="Number of training epochs.",
    )
    ap.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
        help="GNN hidden dimension.",
    )
    ap.add_argument(
        "--num-layers",
        type=int,
        default=3,
        help="Number of GCNConv layers.",
    )
    ap.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate.",
    )
    ap.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay.",
    )
    ap.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout probability.",
    )
    ap.add_argument(
        "--lambda-movement",
        type=float,
        default=0.1,
        help="Weight for Δq movement penalty.",
    )
    ap.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        help="Lightning accelerator (e.g. 'auto', 'cpu', 'gpu').",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    # Datamodule
    dm = KukaTrajGNNDataModule(
        csv_path=args.csv_path,
        batch_size=args.batch_size,
        use_orientation=args.use_orientation,
        val_frac=0.15,
        test_frac=0.15,
        seed=42,
    )
    dm.setup()
    pose_dim = dm.pose_dim
    n_joints = dm.n_joints
    assert pose_dim is not None

    node_input_dim = pose_dim + 2  # pose components + q_prev + joint index scalar

    # Model
    model = IK_GNN(
        node_input_dim=node_input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        output_dim=n_joints,
        dropout=args.dropout,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        lambda_movement=args.lambda_movement,
    )

    # Trainer
    checkpoint_dir = Path("gnn_ik_checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    ckpt_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        dirpath=str(checkpoint_dir),
        filename="gnnik-{epoch:03d}-{val_loss:.4f}",
        save_top_k=1,
        mode="min",
    )
    early_stop = pl.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=20,
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=[ckpt_callback, early_stop],
        accelerator=args.accelerator,
        log_every_n_steps=10,
    )

    trainer.fit(model, dm)
    trainer.test(model, dm)


if __name__ == "__main__":
    main()
