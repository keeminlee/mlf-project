#!/usr/bin/env python3
"""
eval_ik_models.py

Compare trajectory Δq IK models (MLP + GNN) on:
  - joint-space MSE / MAE
  - end-effector (EE) position error via FK in PyBullet
  - average movement magnitude ‖Δq‖ (L1, L2)

Assumptions:
  - CSV is trajectory-mode from kuka_fk_dataset.py --data-type traj
      layout:
        if not include_orientation:
          [ee_x, ee_y, ee_z,
           q_prev_0..6,
           q_curr_0..6]
        if include_orientation:
          [ee_x, ee_y, ee_z, ee_qx, ee_qy, ee_qz, ee_qw,
           q_prev_0..6,
           q_curr_0..6]
  - MLP checkpoint from mlp_ik.py trained with --traj-mode
      (i.e. IKMLP with predict_delta=True)
  - GNN checkpoint from gnn_ik.py trained on the same CSV

Usage example:

  python src/eval_ik_models.py \
    --csv-path data/kuka_traj_dataset_traj.csv \
    --use-orientation \
    --mlp-ckpt mlp_ik_traj_checkpoints/ikmlp-epoch=XXX-val_loss=YYY.ckpt \
    --gnn-ckpt gnn_ik_checkpoints/gnnik-epoch=AAA-val_loss=BBB.ckpt \
    --num-samples 200
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import argparse
import numpy as np
import torch

from torch_geometric.loader import DataLoader as GeoDataLoader

from mlp_ik import IKMLP, load_traj_csv as load_traj_csv_mlp
from gnn_ik import (
    IK_GNN,
    KukaTrajGraphDataset,
    load_traj_csv as load_traj_csv_gnn,
)
from classical_ik import (
    connect_pybullet,
    load_kuka,
    get_revolute_joints,
    fk_position,
)


# --------------------------------------------------------------------------
# FK helpers
# --------------------------------------------------------------------------

def setup_pybullet_fk() -> Tuple[int, int, list[int], int]:
    """Connect to PyBullet in DIRECT mode and load KUKA once."""
    client_id = connect_pybullet(gui=False)
    kuka_uid = load_kuka()
    joint_indices = get_revolute_joints(kuka_uid)
    ee_link_index = joint_indices[-1]
    return client_id, kuka_uid, joint_indices, ee_link_index


def fk_positions_batch(
    kuka_uid: int,
    joint_indices,
    ee_link_index: int,
    q_batch: np.ndarray,
) -> np.ndarray:
    """
    Compute EE positions for a batch of joint configs.

    Args:
        q_batch: (N, 7) array of joint angles

    Returns:
        ee_pos: (N, 3) array of EE xyz positions
    """
    ee_pos = []
    for q in q_batch:
        pos = fk_position(kuka_uid, joint_indices, ee_link_index, q)
        ee_pos.append(pos)
    return np.asarray(ee_pos, dtype=np.float32)


# --------------------------------------------------------------------------
# MLP evaluation (trajectory Δq mode)
# --------------------------------------------------------------------------

def eval_mlp_traj(
    csv_path: str | Path,
    use_orientation: bool,
    ckpt_path: str | Path,
    num_samples: int,
    device: torch.device,
    kuka_uid: int,
    joint_indices,
    ee_link_index: int,
) -> Dict[str, Any]:
    """
    Evaluate MLP IK model trained in trajectory Δq mode.

    Data layout from load_traj_csv_mlp:
        poses:   (N, pose_dim)
        q_prev:  (N, 7)
        q_curr:  (N, 7)

    MLP in traj mode (predict_delta=True) expects:
        X = [pose, q_prev]  -> Δq_pred
        q_pred = q_prev + Δq_pred

    Metrics:
      - joint MSE / MAE vs q_curr
      - EE position MSE / MAE via FK
      - mean ‖Δq‖₂ and ‖Δq‖₁
    """
    csv_path = Path(csv_path)
    ckpt_path = Path(ckpt_path)

    # Load full trajectory dataset
    poses, q_prev, q_curr, pose_dim = load_traj_csv_mlp(
        csv_path, use_orientation=use_orientation
    )
    N = poses.shape[0]
    num_samples = min(num_samples, N)

    rng = np.random.default_rng(0)
    idx = rng.choice(N, size=num_samples, replace=False)

    poses_sub = poses[idx]      # (num_samples, pose_dim)
    q_prev_sub = q_prev[idx]    # (num_samples, 7)
    q_curr_sub = q_curr[idx]    # (num_samples, 7)

    n_joints = q_prev_sub.shape[1]
    assert n_joints == 7, "Expected 7 joint angles for KUKA iiwa."

    # Build MLP inputs: X = [pose, q_prev]
    X_sub = np.concatenate([poses_sub, q_prev_sub], axis=1).astype(np.float32)

    # Load MLP model
    mlp: IKMLP = IKMLP.load_from_checkpoint(ckpt_path)
    mlp.to(device)
    mlp.eval()

    # Check that model was trained in traj Δq mode
    predict_delta = bool(getattr(mlp.hparams, "predict_delta", False))
    if not predict_delta:
        print(
            "[WARNING] Loaded MLP has predict_delta=False. "
            "Treating outputs as absolute q instead of Δq."
        )

    with torch.no_grad():
        x_t = torch.from_numpy(X_sub).to(device)
        out = mlp(x_t)  # shape (num_samples, 7)
        out = out.cpu().numpy().astype(np.float32)

    if predict_delta:
        dq_pred = out
        q_pred = q_prev_sub + dq_pred
    else:
        q_pred = out
        dq_pred = q_pred - q_prev_sub

    # Joint-space metrics
    joint_mse = float(np.mean((q_pred - q_curr_sub) ** 2))
    joint_mae = float(np.mean(np.abs(q_pred - q_curr_sub)))

    # Movement magnitude statistics
    mean_dq_L2 = float(np.mean(np.linalg.norm(dq_pred, axis=1)))
    mean_dq_L1 = float(np.mean(np.sum(np.abs(dq_pred), axis=1)))

    # EE position metrics via FK
    ee_pred = fk_positions_batch(
        kuka_uid, joint_indices, ee_link_index, q_pred
    )  # (N, 3)

    # Target EE: use xyz from pose
    ee_target = poses_sub[:, :3]  # (N, 3)

    ee_mse = float(np.mean((ee_pred - ee_target) ** 2))
    ee_mae = float(np.mean(np.linalg.norm(ee_pred - ee_target, axis=1)))

    return {
        "joint_mse": joint_mse,
        "joint_mae": joint_mae,
        "mean_dq_L2": mean_dq_L2,
        "mean_dq_L1": mean_dq_L1,
        "ee_mse": ee_mse,
        "ee_mae": ee_mae,
        "num_samples": int(num_samples),
    }


# --------------------------------------------------------------------------
# GNN evaluation (trajectory Δq mode)
# --------------------------------------------------------------------------

def eval_gnn_traj(
    csv_path: str | Path,
    use_orientation: bool,
    ckpt_path: str | Path,
    num_samples: int,
    batch_size: int,
    device: torch.device,
    kuka_uid: int,
    joint_indices,
    ee_link_index: int,
) -> Dict[str, Any]:
    """
    Evaluate GNN IK model trained in trajectory Δq mode.

    Uses gnn_ik.load_traj_csv + KukaTrajGraphDataset.

    For each sample:
        data contains pose, q_prev, q_curr encoded as a graph.
        GNN predicts Δq, then:
            q_pred = q_prev + Δq_pred

    Metrics:
      - joint MSE / MAE vs q_curr
      - EE position MSE / MAE via FK
      - mean ‖Δq‖₂ and ‖Δq‖₁
    """
    csv_path = Path(csv_path)
    ckpt_path = Path(ckpt_path)

    # Load full data (we reuse the GNN's loader to avoid layout mismatch)
    poses, q_prev, q_curr, pose_dim = load_traj_csv_gnn(
        csv_path, use_orientation=use_orientation
    )
    N = poses.shape[0]
    num_samples = min(num_samples, N)

    rng = np.random.default_rng(0)
    idx = rng.choice(N, size=num_samples, replace=False)

    poses_sub = poses[idx]
    q_prev_sub = q_prev[idx]
    q_curr_sub = q_curr[idx]

    # Dataset + loader (subset)
    dataset = KukaTrajGraphDataset(poses_sub, q_prev_sub, q_curr_sub, pose_dim)
    loader = GeoDataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Load GNN model from checkpoint
    # Hyperparameters (including node_input_dim) are stored in the checkpoint
    gnn: IK_GNN = IK_GNN.load_from_checkpoint(ckpt_path)
    gnn.to(device)
    gnn.eval()

    all_q_pred = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            dq_pred = gnn(batch)                       # (B, 7)
            q_prev_batch = batch.q_prev.view(-1, 7)    # (B, 7)
            q_pred_batch = q_prev_batch + dq_pred
            all_q_pred.append(q_pred_batch.cpu().numpy().astype(np.float32))

    q_pred = np.concatenate(all_q_pred, axis=0)  # (num_samples, 7)

    # Joint-space metrics
    joint_mse = float(np.mean((q_pred - q_curr_sub) ** 2))
    joint_mae = float(np.mean(np.abs(q_pred - q_curr_sub)))

    dq = q_pred - q_prev_sub
    mean_dq_L2 = float(np.mean(np.linalg.norm(dq, axis=1)))
    mean_dq_L1 = float(np.mean(np.sum(np.abs(dq), axis=1)))

    # EE position metrics via FK
    ee_pred = fk_positions_batch(
        kuka_uid, joint_indices, ee_link_index, q_pred
    )  # (N, 3)
    ee_target = poses_sub[:, :3]

    ee_mse = float(np.mean((ee_pred - ee_target) ** 2))
    ee_mae = float(np.mean(np.linalg.norm(ee_pred - ee_target, axis=1)))

    return {
        "joint_mse": joint_mse,
        "joint_mae": joint_mae,
        "mean_dq_L2": mean_dq_L2,
        "mean_dq_L1": mean_dq_L1,
        "ee_mse": ee_mse,
        "ee_mae": ee_mae,
        "num_samples": int(num_samples),
    }


# --------------------------------------------------------------------------
# CLI + main
# --------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Evaluate trajectory Δq MLP and GNN IK models on KUKA dataset."
    )
    ap.add_argument(
        "--csv-path",
        type=str,
        required=True,
        help="Path to trajectory CSV (from kuka_fk_dataset.py --data-type traj).",
    )
    ap.add_argument(
        "--use-orientation",
        action="store_true",
        help="If set, treat pose as xyz+quat; otherwise xyz.",
    )
    ap.add_argument(
        "--mlp-ckpt",
        type=str,
        default=None,
        help="Path to MLP checkpoint (.ckpt) trained with --traj-mode.",
    )
    ap.add_argument(
        "--gnn-ckpt",
        type=str,
        default=None,
        help="Path to GNN checkpoint (.ckpt).",
    )
    ap.add_argument(
        "--num-samples",
        type=int,
        default=200,
        help="Number of random samples from the CSV for evaluation.",
    )
    ap.add_argument(
        "--gnn-batch-size",
        type=int,
        default=64,
        help="Batch size for GNN evaluation.",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'auto', 'cpu', 'cuda', or 'mps'.",
    )
    return ap.parse_args()


def get_device(arg: str) -> torch.device:
    if arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(arg)


def main() -> None:
    args = parse_args()
    device = get_device(args.device)

    if args.mlp_ckpt is None and args.gnn_ckpt is None:
        raise ValueError("At least one of --mlp-ckpt or --gnn-ckpt must be provided.")

    # Setup PyBullet FK environment once
    client_id, kuka_uid, joint_indices, ee_link_index = setup_pybullet_fk()

    # Evaluate MLP (trajectory Δq)
    if args.mlp_ckpt is not None:
        print("\n=== Evaluating MLP (trajectory Δq) IK model ===")
        mlp_metrics = eval_mlp_traj(
            csv_path=args.csv_path,
            use_orientation=args.use_orientation,
            ckpt_path=args.mlp_ckpt,
            num_samples=args.num_samples,
            device=device,
            kuka_uid=kuka_uid,
            joint_indices=joint_indices,
            ee_link_index=ee_link_index,
        )
        for k, v in mlp_metrics.items():
            if isinstance(v, float):
                print(f"MLP {k:>12}: {v:.6f}")
            else:
                print(f"MLP {k:>12}: {v}")

    # Evaluate GNN (trajectory Δq)
    if args.gnn_ckpt is not None:
        print("\n=== Evaluating GNN (trajectory Δq) IK model ===")
        gnn_metrics = eval_gnn_traj(
            csv_path=args.csv_path,
            use_orientation=args.use_orientation,
            ckpt_path=args.gnn_ckpt,
            num_samples=args.num_samples,
            batch_size=args.gnn_batch_size,
            device=device,
            kuka_uid=kuka_uid,
            joint_indices=joint_indices,
            ee_link_index=ee_link_index,
        )
        for k, v in gnn_metrics.items():
            if isinstance(v, float):
                print(f"GNN {k:>12}: {v:.6f}")
            else:
                print(f"GNN {k:>12}: {v}")

    print("\nDone.")


if __name__ == "__main__":
    main()
