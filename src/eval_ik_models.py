"""
eval_ik_models.py

Compare MLP and GNN IK models (trained on trajectory data) on:
  - joint-space MSE
  - end-effector (EE) position error via FK in PyBullet
  - average joint movement magnitude

Assumes:
  - CSV is trajectory-mode from kuka_fk_dataset.py --data-type traj
  - MLP checkpoint from mlp_ik.py (trained on the same CSV)
  - GNN checkpoint from gnn_ik.py

Usage example:

  python src/eval_ik_models.py \
    --csv-path kuka_traj_dataset_traj.csv \
    --use-orientation \
    --mlp-ckpt mlp_ik_checkpoints/ikmlp-epoch=XXX-val_loss=YYY.ckpt \
    --gnn-ckpt gnn_ik_checkpoints/gnnik-epoch=AAA-val_loss=BBB.ckpt \
    --num-samples 200
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import argparse
import numpy as np
import torch

from torch.utils.data import Subset
from torch_geometric.loader import DataLoader as GeoDataLoader

from mlp_ik import IKMLP, load_ik_csv
from gnn_ik import IK_GNN, KukaTrajGraphDataset, load_traj_csv
from classical_ik import connect_pybullet, load_kuka, get_revolute_joints, fk_position


# --------------------------------------------------------------------------
# FK helpers
# --------------------------------------------------------------------------

def setup_pybullet_fk():
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
# Evaluation logic
# --------------------------------------------------------------------------

def eval_mlp(
    csv_path: str | Path,
    use_orientation: bool,
    ckpt_path: str | Path,
    num_samples: int,
    device: torch.device,
    kuka_uid: int,
    joint_indices,
    ee_link_index: int,
) -> dict:
    """
    Evaluate MLP IK model on a subset of trajectory data.

    MLP in traj mode sees:
      X = [pose, q_prev], y = q_curr

    We compare:
      - joint MSE between q_pred and q_curr
      - EE position error via FK
      - average joint movement magnitude
    """
    csv_path = Path(csv_path)
    ckpt_path = Path(ckpt_path)

    # Load trajectory-style data via load_ik_csv (which auto-detects TRAJ layout)
    X, y, input_dim, n_joints = load_ik_csv(csv_path, use_orientation=use_orientation)
    assert n_joints == 7

    N = X.shape[0]
    num_samples = min(num_samples, N)

    # Random subset
    rng = np.random.default_rng(0)
    idx = rng.choice(N, size=num_samples, replace=False)
    X_sub = X[idx]
    q_curr_sub = y[idx]  # (num_samples, 7)

    # Extract q_prev from X: last 7 dims for traj mode
    q_prev_sub = X_sub[:, -n_joints:]  # (num_samples, 7)

    # Load MLP model from checkpoint
    mlp: IKMLP = IKMLP.load_from_checkpoint(ckpt_path)
    mlp.to(device)
    mlp.eval()

    with torch.no_grad():
        x_t = torch.from_numpy(X_sub).to(device)
        q_pred_t = mlp(x_t)                       # (num_samples, 7)
        q_pred = q_pred_t.cpu().numpy().astype(np.float32)

    # Metrics in joint space
    joint_mse = float(np.mean((q_pred - q_curr_sub) ** 2))
    joint_mae = float(np.mean(np.abs(q_pred - q_curr_sub)))

    # Movement magnitude (how far from q_prev)
    dq = q_pred - q_prev_sub
    mean_dq_L2 = float(np.mean(np.linalg.norm(dq, axis=1)))
    mean_dq_L1 = float(np.mean(np.sum(np.abs(dq), axis=1)))

    # EE position error via FK
    ee_pred = fk_positions_batch(
        kuka_uid, joint_indices, ee_link_index, q_pred
    )  # (N, 3)
    # Target EE is pose part of each row:
    pose_dim = input_dim - n_joints
    poses_full, _, _, _ = load_traj_csv(csv_path, use_orientation=use_orientation)
    ee_target = poses_full[idx, :3]  # xyz

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


def eval_gnn(
    csv_path: str | Path,
    use_orientation: bool,
    ckpt_path: str | Path,
    num_samples: int,
    batch_size: int,
    device: torch.device,
    kuka_uid: int,
    joint_indices,
    ee_link_index: int,
) -> dict:
    """
    Evaluate GNN IK model on a subset of trajectory data.

    GNN sees a small graph per sample and outputs Δq.
    Then:
        q_pred = q_prev + Δq_pred

    We compare:
      - joint MSE between q_pred and q_curr
      - EE position error
      - average step size (Δq)
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

    # Build dataset from subset
    full_dataset = KukaTrajGraphDataset(poses_sub, q_prev_sub, q_curr_sub, pose_dim)
    loader = GeoDataLoader(full_dataset, batch_size=batch_size, shuffle=False)

    # Determine node_input_dim based on dataset
    node_input_dim = full_dataset.node_feat_dim

    # Load GNN model
    gnn: IK_GNN = IK_GNN.load_from_checkpoint(
        ckpt_path,
        node_input_dim=node_input_dim,
        output_dim=7,
    )
    gnn.to(device)
    gnn.eval()

    all_q_pred = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            dq_pred = gnn(batch)              # (B, 7)
            q_prev_batch = batch.q_prev.view(-1, 7)  # (B, 7)
            q_pred_batch = q_prev_batch + dq_pred
            all_q_pred.append(q_pred_batch.cpu().numpy())

    q_pred = np.concatenate(all_q_pred, axis=0)  # (num_samples, 7)

    # Joint-space metrics
    joint_mse = float(np.mean((q_pred - q_curr_sub) ** 2))
    joint_mae = float(np.mean(np.abs(q_pred - q_curr_sub)))

    dq = q_pred - q_prev_sub
    mean_dq_L2 = float(np.mean(np.linalg.norm(dq, axis=1)))
    mean_dq_L1 = float(np.mean(np.sum(np.abs(dq), axis=1)))

    # EE metrics via FK
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
        description="Evaluate MLP and GNN IK models on trajectory KUKA dataset."
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
        help="Path to MLP checkpoint (.ckpt). If omitted, MLP is skipped.",
    )
    ap.add_argument(
        "--gnn-ckpt",
        type=str,
        default=None,
        help="Path to GNN checkpoint (.ckpt). If omitted, GNN is skipped.",
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

    # Evaluate MLP
    if args.mlp_ckpt is not None:
        print("\n=== Evaluating MLP IK model ===")
        mlp_metrics = eval_mlp(
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
            print(f"MLP {k:>12}: {v:.6f}" if isinstance(v, float) else f"MLP {k:>12}: {v}")

    # Evaluate GNN
    if args.gnn_ckpt is not None:
        print("\n=== Evaluating GNN IK model ===")
        gnn_metrics = eval_gnn(
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
            print(f"GNN {k:>12}: {v:.6f}" if isinstance(v, float) else f"GNN {k:>12}: {v}")

    print("\nDone.")


if __name__ == "__main__":
    main()
