#!/usr/bin/env python3
"""
mlp_ik.py

MLP baseline for learning inverse kinematics (IK) for the KUKA iiwa arm.

Modes:
- Single-shot mode (default):
    Input:  end-effector pose (xyz or xyz+quat)
    Output: joint angles (7D)
    Loss:   MSE on joint angles

- Trajectory Δq mode (--traj-mode):
    CSV from kuka_fk_dataset.py --data-type traj:
      [pose, prev_joints, curr_joints]
    Input:  [pose, q_prev]
    Output: Δq
    Loss:   ik_loss    = MSE(q_prev + Δq_pred, q_curr)
            move_loss  = MSE(Δq_pred, 0)
            total_loss = ik_loss + lambda_movement * move_loss

This script provides:
- IKMLP:           LightningModule for IK regression
- KukaIKDataModule:      single-shot CSV dataset
- KukaIKTrajDataModule:  trajectory Δq dataset
- build_ik_mlp_model / build_ik_datamodule: factory functions (single-shot)
- run_mlp_grid_search:   convenience wrapper using IKGridSearch (single-shot)
- main(): CLI for single-shot or trajectory Δq training
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import argparse
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl

# If grid_search.py is in the same src/ directory:
from grid_search import IKGridSearch


# ------------------------------------------------------------------------------
# Data loading utilities
# ------------------------------------------------------------------------------

def load_ik_csv(
    csv_path: str | Path,
    use_orientation: bool = False,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """
    Load IK dataset from a CSV produced by kuka_fk_dataset.py in single-shot mode.

    The CSV columns are:
      - if not include_orientation:
            [ee_x, ee_y, ee_z, joint_0, ..., joint_6]
      - if include_orientation:
            [ee_x, ee_y, ee_z, ee_qx, ee_qy, ee_qz, ee_qw, joint_0, ..., joint_6]

    Args:
        csv_path: path to the CSV file
        use_orientation: if True, expect (and use) 7D pose; else use only xyz

    Returns:
        X:          (N, pose_dim)  - poses
        y:          (N, n_joints)  - joint angles
        pose_dim:   int
        n_joints:   int
    """
    csv_path = Path(csv_path)
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)

    pose_dim = 7 if use_orientation else 3
    X = data[:, :pose_dim]
    y = data[:, pose_dim:]

    n_joints = y.shape[1]

    return X.astype(np.float32), y.astype(np.float32), pose_dim, n_joints


def load_traj_csv(
    csv_path: str | Path,
    use_orientation: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Load trajectory-mode CSV produced by kuka_fk_dataset.py with --data-type traj.

    Layout (include_orientation=False):
        [ee_x, ee_y, ee_z,
         prev_joint_0..6,
         curr_joint_0..6]

    Layout (include_orientation=True):
        [ee_x, ee_y, ee_z, ee_qx, ee_qy, ee_qz, ee_qw,
         prev_joint_0..6,
         curr_joint_0..6]

    Returns:
        poses:    (N, pose_dim)
        q_prev:   (N, 7)
        q_curr:   (N, 7)
        pose_dim: int
    """
    csv_path = Path(csv_path)
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1).astype(np.float32)

    pose_dim = 7 if use_orientation else 3
    pose = data[:, :pose_dim]
    q_prev = data[:, pose_dim:pose_dim + 7]
    q_curr = data[:, pose_dim + 7:pose_dim + 14]
    return pose, q_prev, q_curr, pose_dim


# ------------------------------------------------------------------------------
# Lightning DataModules
# ------------------------------------------------------------------------------

class KukaIKDataModule(pl.LightningDataModule):
    """Data module for KUKA IK MLP training from a single-shot CSV."""

    def __init__(
        self,
        csv_path: str | Path,
        batch_size: int = 256,
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

        # Will be set in setup()
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.X_val: Optional[np.ndarray] = None
        self.y_val: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None

        self.pose_dim: Optional[int] = None
        self.n_joints: Optional[int] = None

    def setup(self, stage: Optional[str] = None):
        # Load full dataset
        X, y, pose_dim, n_joints = load_ik_csv(self.csv_path, use_orientation=self.use_orientation)
        self.pose_dim = pose_dim
        self.n_joints = n_joints

        N = len(X)
        rng = np.random.default_rng(self.seed)
        indices = np.arange(N)
        rng.shuffle(indices)

        n_val = int(self.val_frac * N)
        n_test = int(self.test_frac * N)
        n_train = N - n_val - n_test

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        self.X_train, self.y_train = X[train_idx], y[train_idx]
        self.X_val, self.y_val = X[val_idx], y[val_idx]
        self.X_test, self.y_test = X[test_idx], y[test_idx]

        self.train_dataset = TensorDataset(
            torch.from_numpy(self.X_train),
            torch.from_numpy(self.y_train),
        )
        self.val_dataset = TensorDataset(
            torch.from_numpy(self.X_val),
            torch.from_numpy(self.y_val),
        )
        self.test_dataset = TensorDataset(
            torch.from_numpy(self.X_test),
            torch.from_numpy(self.y_test),
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)


class KukaIKTrajDataModule(pl.LightningDataModule):
    """
    Trajectory Δq data module.

    Each sample:
      X = [pose, q_prev]
      y = q_curr

    The model in traj mode predicts Δq; q_prev is recovered from X inside the
    LightningModule.
    """

    def __init__(
        self,
        csv_path: str | Path,
        batch_size: int = 256,
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
        self.n_joints: Optional[int] = None

    def setup(self, stage: Optional[str] = None):
        poses, q_prev, q_curr, pose_dim = load_traj_csv(
            self.csv_path,
            use_orientation=self.use_orientation,
        )
        self.pose_dim = pose_dim
        self.n_joints = q_prev.shape[1]

        # Build X = [pose, q_prev], y = q_curr
        X = np.concatenate([poses, q_prev], axis=1).astype(np.float32)
        y = q_curr.astype(np.float32)

        N = len(X)
        rng = np.random.default_rng(self.seed)
        indices = np.arange(N)
        rng.shuffle(indices)

        n_val = int(self.val_frac * N)
        n_test = int(self.test_frac * N)
        n_train = N - n_val - n_test

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        self.train_dataset = TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(y_train),
        )
        self.val_dataset = TensorDataset(
            torch.from_numpy(X_val),
            torch.from_numpy(y_val),
        )
        self.test_dataset = TensorDataset(
            torch.from_numpy(X_test),
            torch.from_numpy(y_test),
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)


# ------------------------------------------------------------------------------
# Lightning MLP model for IK
# ------------------------------------------------------------------------------

class IKMLP(pl.LightningModule):
    """
    MLP for inverse kinematics.

    Modes:
      - predict_delta=False (single-shot):
          x = pose
          y = q
          loss = MSE(q_pred, q)

      - predict_delta=True (trajectory Δq):
          x = [pose, q_prev]
          y = q_curr
          dq_pred = mlp(x)
          q_pred  = q_prev + dq_pred
          ik_loss   = MSE(q_pred, q_curr)
          move_loss = MSE(dq_pred, 0)
          loss      = ik_loss + lambda_movement * move_loss
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 7,
        hidden_dims: List[int] = [256, 256, 128],
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        predict_delta: bool = False,
        lambda_movement: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()

        layers: List[nn.Module] = []
        prev_dim = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.output_dim = output_dim
        self.input_dim = input_dim

        self.predict_delta = predict_delta
        self.lambda_movement = lambda_movement

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim)

        Returns:
            (B, output_dim)
              - If predict_delta=False: predicted joints q_pred
              - If predict_delta=True:  predicted Δq
        """
        return self.mlp(x)

    # ---- helpers for traj mode ----

    def _split_prev_curr_from_batch(self, x: torch.Tensor, y: torch.Tensor):
        """
        For trajectory mode, recover q_prev from the last output_dim entries in x,
        and q_curr from y.
        """
        n_joints = self.output_dim
        q_prev = x[:, self.input_dim - n_joints:]
        q_curr = y
        return q_prev, q_curr

    def _traj_losses(self, x: torch.Tensor, y: torch.Tensor, dq_pred: torch.Tensor):
        """
        Compute IK + movement losses in trajectory mode.
        """
        q_prev, q_curr = self._split_prev_curr_from_batch(x, y)
        q_pred = q_prev + dq_pred

        ik_loss = F.mse_loss(q_pred, q_curr)
        move_loss = F.mse_loss(dq_pred, torch.zeros_like(dq_pred))
        loss = ik_loss + self.lambda_movement * move_loss
        return ik_loss, move_loss, loss

    # ---- training / validation / test steps ----

    def training_step(self, batch, batch_idx: int):
        x, y = batch  # x: input, y: target

        if self.predict_delta:
            dq_pred = self(x)
            ik_loss, move_loss, loss = self._traj_losses(x, y, dq_pred)
            self.log("train_ik_loss", ik_loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log("train_move_loss", move_loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        else:
            y_pred = self(x)
            loss = F.mse_loss(y_pred, y)
            self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx: int):
        x, y = batch

        if self.predict_delta:
            dq_pred = self(x)
            ik_loss, move_loss, loss = self._traj_losses(x, y, dq_pred)
            self.log("val_ik_loss", ik_loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log("val_move_loss", move_loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
            return loss
        else:
            y_pred = self(x)
            loss = F.mse_loss(y_pred, y)
            mae = F.l1_loss(y_pred, y)
            self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log("val_mae", mae, prog_bar=True, on_step=False, on_epoch=True)
            return loss

    def test_step(self, batch, batch_idx: int):
        x, y = batch

        if self.predict_delta:
            dq_pred = self(x)
            ik_loss, move_loss, loss = self._traj_losses(x, y, dq_pred)
            self.log("test_ik_loss", ik_loss, prog_bar=False)
            self.log("test_move_loss", move_loss, prog_bar=False)
            self.log("test_loss", loss, prog_bar=False)
            return loss
        else:
            y_pred = self(x)
            loss = F.mse_loss(y_pred, y)
            mae = F.l1_loss(y_pred, y)
            self.log("test_loss", loss, prog_bar=False)
            self.log("test_mae", mae, prog_bar=False)
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
# Factories for IKGridSearch (single-shot mode)
# ------------------------------------------------------------------------------

def build_ik_mlp_model(params: Dict[str, Any], n_joints: int) -> pl.LightningModule:
    """
    Factory function for IKGridSearch (single-shot mode).

    Args:
        params: hyperparameters for this run (hidden_dims, dropout, learning_rate,
                weight_decay, use_orientation)
        n_joints: number of joints (7 for KUKA iiwa)

    Returns:
        IKMLP LightningModule
    """
    pose_dim = 7 if params.get("use_orientation", False) else 3

    return IKMLP(
        input_dim=pose_dim,
        output_dim=n_joints,
        hidden_dims=params["hidden_dims"],
        dropout=params.get("dropout", 0.1),
        learning_rate=params.get("learning_rate", 1e-3),
        weight_decay=params.get("weight_decay", 1e-4),
        predict_delta=False,
    )


def build_ik_datamodule(params: Dict[str, Any], splits: Dict[str, Any]) -> pl.LightningDataModule:
    """
    Factory function for IKGridSearch (single-shot mode).

    Args:
        params: hyperparameters for this run (e.g., batch_size, use_orientation)
        splits: config dict describing dataset / split (csv_path, val_frac, test_frac, seed)

    Returns:
        KukaIKDataModule
    """
    csv_path = splits["csv_path"]
    use_orientation = params.get("use_orientation", splits.get("use_orientation", False))

    return KukaIKDataModule(
        csv_path=csv_path,
        batch_size=params.get("batch_size", 256),
        use_orientation=use_orientation,
        val_frac=splits.get("val_frac", 0.15),
        test_frac=splits.get("test_frac", 0.15),
        seed=splits.get("seed", 42),
    )


def run_mlp_grid_search(
    splits: Dict[str, Any],
    output_dir: str | Path,
    max_epochs: int = 100,
    n_joints: int = 7,
) -> Dict[str, Any]:
    """
    Convenience wrapper to run grid search for the IK MLP (single-shot mode).

    Args:
        splits: dict with dataset/splitting info (must include 'csv_path')
        output_dir: directory where grid search artifacts go
        max_epochs: max epochs per config
        n_joints: number of joints (7 for KUKA iiwa)

    Returns:
        best_result dict: { "params", "best_score", "best_path" }
    """
    param_grid: Dict[str, List[Any]] = {
        "hidden_dims": [
            [256, 256],
            [256, 256, 128],
            [512, 256, 128],
        ],
        "dropout": [0.0, 0.1, 0.2],
        "learning_rate": [1e-3, 5e-4],
        "weight_decay": [1e-4, 1e-5],
        "use_orientation": [False, True],
        "batch_size": [128, 256],
    }

    runner = IKGridSearch(
        output_dir=output_dir,
        param_grid=param_grid,
        model_builder=build_ik_mlp_model,
        datamodule_builder=build_ik_datamodule,
        monitor_metric="val_loss",
        mode="min",
        patience=15,
        max_epochs=max_epochs,
        accelerator="auto",
        n_joints=n_joints,
    )

    results = runner.run(splits)
    if not results:
        raise RuntimeError("No successful IK MLP runs in grid search.")
    best = min(results, key=lambda r: r["best_score"])
    return best


# ------------------------------------------------------------------------------
# Simple single-run training CLI
# ------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train an MLP IK baseline on KUKA FK dataset.")
    ap.add_argument(
        "--csv-path",
        type=str,
        default="data/kuka_fk_dataset.csv",
        help="Path to FK-generated CSV (from kuka_fk_dataset.py).",
    )
    ap.add_argument(
        "--use-orientation",
        action="store_true",
        help="If set, use xyz+quat (7D pose) instead of xyz (3D).",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for training.",
    )
    ap.add_argument(
        "--max-epochs",
        type=int,
        default=100,
        help="Number of training epochs.",
    )
    ap.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[256, 256, 128],
        help="Hidden layer sizes, e.g. --hidden-dims 256 256 128",
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
        "--accelerator",
        type=str,
        default="auto",
        help="Lightning accelerator argument (e.g. 'auto', 'cpu', 'gpu', 'mps').",
    )
    ap.add_argument(
        "--traj-mode",
        action="store_true",
        help="If set, train in trajectory Δq mode on a traj CSV from kuka_fk_dataset.py.",
    )
    ap.add_argument(
        "--lambda-movement",
        type=float,
        default=0.1,
        help="Movement penalty weight in trajectory mode.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if args.traj_mode:
        # Trajectory Δq mode
        dm = KukaIKTrajDataModule(
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
        assert pose_dim is not None and n_joints is not None

        input_dim = pose_dim + n_joints  # [pose, q_prev]
        model = IKMLP(
            input_dim=input_dim,
            output_dim=n_joints,
            hidden_dims=args.hidden_dims,
            dropout=args.dropout,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            predict_delta=True,
            lambda_movement=args.lambda_movement,
        )
        checkpoint_dir = Path("mlp_ik_traj_checkpoints")
    else:
        # Single-shot mode
        dm = KukaIKDataModule(
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
        assert pose_dim is not None and n_joints is not None

        model = IKMLP(
            input_dim=pose_dim,
            output_dim=n_joints,
            hidden_dims=args.hidden_dims,
            dropout=args.dropout,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            predict_delta=False,
        )
        checkpoint_dir = Path("mlp_ik_checkpoints")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    ckpt_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        dirpath=str(checkpoint_dir),
        filename="ikmlp-{epoch:03d}-{val_loss:.4f}",
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
