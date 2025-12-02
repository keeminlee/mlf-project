#!/usr/bin/env python3
"""
eval_ik_models.py

Compare IK models on:
  - joint-space MSE / MAE
  - end-effector (EE) position error via FK in PyBullet
  - average movement magnitude ‖Δq‖ (L1, L2)

Supports:
  - MLP trained in single-shot mode (pose -> q)
  - MLP trained in trajectory Δq mode ( [pose, q_prev] -> Δq or q_curr )
  - GNN trained in trajectory Δq mode

Usage example (traj models):

  python src/eval_ik_models.py \
    --csv-path data/kuka_traj_dataset_traj.csv \
    --use-orientation \
    --mlp-ckpt mlp_ik_traj_checkpoints/ikmlp-epoch=XXX-val_loss=YYY.ckpt \
    --gnn-ckpt gnn_ik_checkpoints/gnnik-epoch=AAA-val_loss=BBB.ckpt \
    --num-samples 200

Usage example (base single-shot MLP only):

  python src/eval_ik_models.py \
    --csv-path data/kuka_fk_dataset.csv \
    --use-orientation \
    --mlp-ckpt mlp_ik_checkpoints/ikmlp-epoch=ZZZ-val_loss=WWW.ckpt \
    --num-samples 200
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple

import argparse
import numpy as np
import torch

from torch_geometric.loader import DataLoader as GeoDataLoader

from mlp_ik import IKMLP
from gnn_ik import (
    IK_GNN,
    KukaTrajGraphDataset,
)
from data_utils import load_ik_csv, load_traj_csv
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
# MLP evaluation: single-shot mode
# --------------------------------------------------------------------------

def eval_mlp_single(
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
    Evaluate MLP IK model trained in single-shot mode.

    Data layout from load_ik_csv:
        X: (N, pose_dim)
        y: (N, 7)  joint angles

    MLP behavior:
        x = pose
        y_pred = q_pred

    Metrics:
      - joint MSE / MAE vs q
      - EE position MSE / MAE via FK
      - movement ‖Δq‖ relative to some reference (we'll compute vs true q for consistency)
    """
    csv_path = Path(csv_path)
    ckpt_path = Path(ckpt_path)

    X, y, pose_dim, n_joints = load_ik_csv(csv_path, use_orientation=use_orientation)
    assert n_joints == 7
    N = X.shape[0]
    num_samples = min(num_samples, N)

    rng = np.random.default_rng(0)
    idx = rng.choice(N, size=num_samples, replace=False)

    X_sub = X[idx]          # poses
    q_true_sub = y[idx]     # ground-truth joints

    mlp: IKMLP = IKMLP.load_from_checkpoint(ckpt_path)
    mlp.to(device)
    mlp.eval()

    with torch.no_grad():
        x_t = torch.from_numpy(X_sub).to(device)
        q_pred_t = mlp(x_t)
        q_pred = q_pred_t.cpu().numpy().astype(np.float32)

    # Joint-space metrics
    joint_mse = float(np.mean((q_pred - q_true_sub) ** 2))
    joint_mae = float(np.mean(np.abs(q_pred - q_true_sub)))

    # "Movement" here is just ‖q_pred - q_true‖; there is no q_prev in single-shot
    dq = q_pred - q_true_sub
    mean_dq_L2 = float(np.mean(np.linalg.norm(dq, axis=1)))
    mean_dq_L1 = float(np.mean(np.sum(np.abs(dq), axis=1)))

    # EE metrics via FK
    ee_pred = fk_positions_batch(
        kuka_uid, joint_indices, ee_link_index, q_pred
    )
    ee_target = X_sub[:, :3]  # xyz part of pose

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
        "mode": "single",
    }


# --------------------------------------------------------------------------
# MLP evaluation: trajectory Δq mode
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
    Evaluate MLP IK model trained in trajectory mode.

    Data layout from load_traj_csv:
        poses:   (N, pose_dim)
        q_prev:  (N, 7)
        q_curr:  (N, 7)

    MLP behavior (Δq or q_curr depending on predict_delta):
        x = [pose, q_prev] or [pose] (if misconfigured)
        out:
          - if predict_delta=True: Δq_pred
          - else: q_pred

    Metrics:
      - joint MSE / MAE vs q_curr
      - EE position MSE / MAE via FK
      - movement ‖Δq‖
    """
    csv_path = Path(csv_path)
    ckpt_path = Path(ckpt_path)

    poses, q_prev, q_curr, pose_dim = load_traj_csv(
        csv_path, use_orientation=use_orientation
    )
    N = poses.shape[0]
    num_samples = min(num_samples, N)

    rng = np.random.default_rng(0)
    idx = rng.choice(N, size=num_samples, replace=False)

    poses_sub = poses[idx]
    q_prev_sub = q_prev[idx]
    q_curr_sub = q_curr[idx]

    n_joints = q_prev_sub.shape[1]
    assert n_joints == 7

    mlp: IKMLP = IKMLP.load_from_checkpoint(ckpt_path)
    mlp.to(device)
    mlp.eval()

    predict_delta = bool(getattr(mlp.hparams, "predict_delta", False))
    input_dim = int(getattr(mlp.hparams, "input_dim", pose_dim))

    if input_dim == pose_dim + 7:
        X_sub = np.concatenate([poses_sub, q_prev_sub], axis=1).astype(np.float32)
    elif input_dim == pose_dim:
        # Slightly odd but we can still try: no q_prev included
        X_sub = poses_sub.astype(np.float32)
    else:
        raise ValueError(
            f"Unexpected MLP input_dim={input_dim}, pose_dim={pose_dim} for trajectory data."
        )

    with torch.no_grad():
        x_t = torch.from_numpy(X_sub).to(device)
        out = mlp(x_t).cpu().numpy().astype(np.float32)

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
    )
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
        "mode": "traj",
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
    """
    csv_path = Path(csv_path)
    ckpt_path = Path(ckpt_path)

    poses, q_prev, q_curr, pose_dim = load_traj_csv(
        csv_path, use_orientation=use_orientation
    )
    N = poses.shape[0]
    num_samples = min(num_samples, N)

    rng = np.random.default_rng(0)
    idx = rng.choice(N, size=num_samples, replace=False)

    poses_sub = poses[idx]
    q_prev_sub = q_prev[idx]
    q_curr_sub = q_curr[idx]

    dataset = KukaTrajGraphDataset(poses_sub, q_prev_sub, q_curr_sub, pose_dim)
    loader = GeoDataLoader(dataset, batch_size=batch_size, shuffle=False)

    gnn: IK_GNN = IK_GNN.load_from_checkpoint(ckpt_path)
    gnn.to(device)
    gnn.eval()

    all_q_pred = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            dq_pred = gnn(batch)
            q_prev_batch = batch.q_prev.view(-1, 7)
            q_pred_batch = q_prev_batch + dq_pred
            all_q_pred.append(q_pred_batch.cpu().numpy().astype(np.float32))

    q_pred = np.concatenate(all_q_pred, axis=0)

    joint_mse = float(np.mean((q_pred - q_curr_sub) ** 2))
    joint_mae = float(np.mean(np.abs(q_pred - q_curr_sub)))

    dq = q_pred - q_prev_sub
    mean_dq_L2 = float(np.mean(np.linalg.norm(dq, axis=1)))
    mean_dq_L1 = float(np.mean(np.sum(np.abs(dq), axis=1)))

    ee_pred = fk_positions_batch(
        kuka_uid, joint_indices, ee_link_index, q_pred
    )
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
        description="Evaluate MLP/GNN IK models on KUKA datasets."
    )
    ap.add_argument(
        "--csv-path",
        type=str,
        required=True,
        help="Path to CSV. For single-shot MLP: kuka_fk_dataset.csv; "
             "for traj models: kuka_traj_dataset_traj.csv.",
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
        help="Path to MLP checkpoint (.ckpt). Auto-detects single vs traj via predict_delta.",
    )
    ap.add_argument(
        "--gnn-ckpt",
        type=str,
        default=None,
        help="Path to GNN checkpoint (.ckpt). Requires traj CSV.",
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

    client_id, kuka_uid, joint_indices, ee_link_index = setup_pybullet_fk()

    # --- MLP evaluation (auto single vs traj) ---
    if args.mlp_ckpt is not None:
        mlp: IKMLP = IKMLP.load_from_checkpoint(args.mlp_ckpt)
        predict_delta = bool(getattr(mlp.hparams, "predict_delta", False))
        mlp.cpu()  # we reload inside each eval on the proper device anyway

        print("\n=== Evaluating MLP IK model ===")
        if predict_delta:
            print("[MLP] Detected trajectory Δq mode (predict_delta=True)")
            metrics = eval_mlp_traj(
                csv_path=args.csv_path,
                use_orientation=args.use_orientation,
                ckpt_path=args.mlp_ckpt,
                num_samples=args.num_samples,
                device=device,
                kuka_uid=kuka_uid,
                joint_indices=joint_indices,
                ee_link_index=ee_link_index,
            )
        else:
            print("[MLP] Detected single-shot mode (predict_delta=False)")
            metrics = eval_mlp_single(
                csv_path=args.csv_path,
                use_orientation=args.use_orientation,
                ckpt_path=args.mlp_ckpt,
                num_samples=args.num_samples,
                device=device,
                kuka_uid=kuka_uid,
                joint_indices=joint_indices,
                ee_link_index=ee_link_index,
            )

        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"MLP {k:>12}: {v:.6f}")
            else:
                print(f"MLP {k:>12}: {v}")

    # --- GNN evaluation (traj only) ---
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
