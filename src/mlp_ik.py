#!/usr/bin/env python3
"""
mlp_ik.py

MLP baseline for learning inverse kinematics (IK) for the KUKA iiwa arm.

- Input:  end-effector pose (xyz or xyz+quat)
- Output: joint angles (7D)
- Model:  fully-connected MLP (no residual), trained with MSE on joint angles

This script provides:
- IKMLP:          LightningModule for IK regression
- KukaIKDataModule: LightningDataModule for CSV-based dataset
- build_ik_mlp_model / build_ik_datamodule: factory functions for grid search
- run_mlp_grid_search: convenience wrapper using IKGridSearch
- main(): simple single-run training with fixed hyperparameters
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import argparse
import json
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
    Load IK dataset from a CSV produced by kuka_fk_dataset.py.

    The CSV columns are:
      - if not include_orientation: [ee_x, ee_y, ee_z, joint_0, ..., joint_6]
      - if include_orientation:     [ee_x, ee_y, ee_z, ee_qx, ee_qy, ee_qz, ee_qw, joint_0, ..., joint_6]

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


# ------------------------------------------------------------------------------
# Lightning DataModule
# ------------------------------------------------------------------------------

class KukaIKDataModule(pl.LightningDataModule):
    """Data module for KUKA IK MLP training from a single CSV."""

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


# ------------------------------------------------------------------------------
# Lightning MLP model for IK
# ------------------------------------------------------------------------------

class IKMLP(pl.LightningModule):
    """Plain MLP for inverse kinematics: pose -> joint angles."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 7,
        hidden_dims: List[int] = [256, 256, 128],
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim) pose vector

        Returns:
            (B, output_dim) predicted joint angles
        """
        return self.mlp(x)

    def training_step(self, batch, batch_idx: int):
        x, y = batch  # x: pose, y: joint angles
        y_pred = self(x)
        loss = F.mse_loss(y_pred, y)

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx: int):
        x, y = batch
        y_pred = self(x)
        loss = F.mse_loss(y_pred, y)
        # mean absolute error can be nice to track
        mae = F.l1_loss(y_pred, y)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_mae", mae, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx: int):
        x, y = batch
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
# Factories for IKGridSearch
# ------------------------------------------------------------------------------

def build_ik_mlp_model(params: Dict[str, Any], n_joints: int) -> pl.LightningModule:
    """
    Factory function for IKGridSearch.

    Args:
        params: hyperparameters for this run (hidden_dims, dropout, learning_rate, weight_decay, use_orientation)
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
    )


def build_ik_datamodule(params: Dict[str, Any], splits: Dict[str, Any]) -> pl.LightningDataModule:
    """
    Factory function for IKGridSearch.

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
    Convenience wrapper to run grid search for the IK MLP.

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
        default="../data/kuka_fk_dataset.csv",
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
        help="Lightning accelerator argument (e.g. 'auto', 'cpu', 'gpu').",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    # Data
    datamodule = KukaIKDataModule(
        csv_path=args.csv_path,
        batch_size=args.batch_size,
        use_orientation=args.use_orientation,
        val_frac=0.15,
        test_frac=0.15,
        seed=42,
    )
    datamodule.setup()
    pose_dim = datamodule.pose_dim
    n_joints = datamodule.n_joints
    assert pose_dim is not None and n_joints is not None

    # Model
    model = IKMLP(
        input_dim=pose_dim,
        output_dim=n_joints,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
    )

    # Trainer
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

    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)


if __name__ == "__main__":
    main()
