#!/usr/bin/env python3
import argparse
import math
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_indices(num_samples: int, val_ratio: float, test_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(num_samples)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    test_size = int(num_samples * test_ratio)
    val_size = int(num_samples * val_ratio)

    test_idx = indices[:test_size]
    val_idx = indices[test_size:test_size + val_size]
    train_idx = indices[test_size + val_size:]
    return train_idx, val_idx, test_idx


def standardize(train_arr: np.ndarray, other_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    mean = train_arr.mean(axis=0, keepdims=True)
    std = train_arr.std(axis=0, keepdims=True) + 1e-8

    def transform(arr: np.ndarray) -> np.ndarray:
        return (arr - mean) / std

    meta = {"mean": mean, "std": std}
    return transform(train_arr), transform(other_arr), meta


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 256, depth: int = 3, dropout: float = 0.1):
        super().__init__()
        layers = []
        last_dim = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_dataloaders(features: np.ndarray, targets: np.ndarray, batch_size: int, val_ratio: float, test_ratio: float, seed: int) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    train_idx, val_idx, test_idx = split_indices(len(features), val_ratio, test_ratio, seed)

    X_train, X_val, X_test = features[train_idx], features[val_idx], features[test_idx]
    y_train, y_val, y_test = targets[train_idx], targets[val_idx], targets[test_idx]

    X_train_norm, _, x_stats = standardize(X_train, X_train)
    X_val_norm = (X_val - x_stats["mean"]) / x_stats["std"]
    X_test_norm = (X_test - x_stats["mean"]) / x_stats["std"]

    y_train_norm, _, y_stats = standardize(y_train, y_train)
    y_val_norm = (y_val - y_stats["mean"]) / y_stats["std"]
    y_test_norm = (y_test - y_stats["mean"]) / y_stats["std"]

    def make_loader(x_arr: np.ndarray, y_arr: np.ndarray, shuffle: bool) -> DataLoader:
        dataset = TensorDataset(torch.from_numpy(x_arr).float(), torch.from_numpy(y_arr).float())
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    train_loader = make_loader(X_train_norm, y_train_norm, shuffle=True)
    val_loader = make_loader(X_val_norm, y_val_norm, shuffle=False)
    test_loader = make_loader(X_test_norm, y_test_norm, shuffle=False)

    return train_loader, val_loader, test_loader, x_stats, y_stats, (train_idx, val_idx, test_idx)


def invert_standardization(tensor: torch.Tensor, stats: Dict[str, np.ndarray]) -> torch.Tensor:
    mean = torch.from_numpy(stats["mean"]).to(tensor.device).float()
    std = torch.from_numpy(stats["std"]).to(tensor.device).float()
    return tensor * std + mean


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, y_stats: Dict[str, np.ndarray]) -> Dict[str, float]:
    model.eval()
    mse_loss = nn.MSELoss(reduction="sum")
    mae_loss = nn.L1Loss(reduction="sum")

    total_mse = 0.0
    total_mae = 0.0
    total_samples = 0

    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)

            preds = model(features)
            preds_denorm = invert_standardization(preds, y_stats)
            targets_denorm = invert_standardization(targets, y_stats)

            total_mse += mse_loss(preds_denorm, targets_denorm).item()
            total_mae += mae_loss(preds_denorm, targets_denorm).item()
            total_samples += targets.size(0)

    mse = total_mse / total_samples
    mae = total_mae / total_samples
    rmse = math.sqrt(mse)
    return {"mse": mse, "mae": mae, "rmse": rmse}


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    set_seed(args.seed)

    df = pd.read_csv(args.data_path)
    feature_cols = ["ee_x", "ee_y", "ee_z", "ee_qx", "ee_qy", "ee_qz", "ee_qw"]
    target_cols = [f"joint_{i}" for i in range(7)]

    features = df[feature_cols].to_numpy(dtype=np.float32)
    targets = df[target_cols].to_numpy(dtype=np.float32)

    train_loader, val_loader, test_loader, x_stats, y_stats, splits = build_dataloaders(
        features,
        targets,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    model = MLP(in_dim=features.shape[1], out_dim=len(target_cols), hidden_dim=args.hidden_dim, depth=args.depth, dropout=args.dropout)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for features_batch, targets_batch in train_loader:
            features_batch = features_batch.to(device)
            targets_batch = targets_batch.to(device)

            preds = model(features_batch)
            loss = criterion(preds, targets_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * features_batch.size(0)

        epoch_loss /= len(train_loader.dataset)
        val_metrics = evaluate(model, val_loader, device, y_stats)

        if val_metrics["rmse"] < best_val:
            best_val = val_metrics["rmse"]
            best_state = model.state_dict()

        if epoch % args.log_interval == 0 or epoch == 1 or epoch == args.epochs:
            print(
                f"Epoch {epoch:03d} | Train Loss: {epoch_loss:.4f} | "
                f"Val RMSE: {val_metrics['rmse']:.4f} | Val MAE: {val_metrics['mae']:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    train_metrics = evaluate(model, train_loader, device, y_stats)
    val_metrics = evaluate(model, val_loader, device, y_stats)
    test_metrics = evaluate(model, test_loader, device, y_stats)

    print("\nFinal performance (denormalized joint units):")
    for split_name, metrics in [("Train", train_metrics), ("Validation", val_metrics), ("Test", test_metrics)]:
        print(
            f"{split_name}: RMSE={metrics['rmse']:.4f}, MSE={metrics['mse']:.4f}, MAE={metrics['mae']:.4f}"
        )

    if args.save_path:
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "feature_stats": x_stats,
                "target_stats": y_stats,
                "feature_cols": feature_cols,
                "target_cols": target_cols,
                "splits": splits,
                "args": vars(args),
            },
            save_path,
        )
        print(f"\nModel checkpoint saved to {save_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an MLP inverse kinematics regressor for KUKA arm.")
    parser.add_argument("--data-path", type=str, default="kuka_fk_pose7d.csv", help="CSV dataset path.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size.")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs.")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden layer width.")
    parser.add_argument("--depth", type=int, default=3, help="Number of hidden layers.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay for Adam.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    parser.add_argument("--save-path", type=str, default="", help="Optional path to store trained checkpoint.")
    parser.add_argument("--log-interval", type=int, default=10, help="Epoch interval for logging.")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())

