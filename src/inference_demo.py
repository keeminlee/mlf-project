#!/usr/bin/env python3
"""
inference_demo.py

Simple inference demo showing: "Given a target pose, here's the predicted trajectory."

This script demonstrates how to use trained MLP and GNN models to predict
joint configurations for a sequence of target end-effector poses.

Example usage:
  python src/inference_demo.py \
    --csv-path data/kuka_traj_dataset_traj.csv \
    --use-orientation \
    --num-steps 10
"""

from __future__ import annotations

import argparse
import numpy as np
import torch
from pathlib import Path

from checkpoint_utils import auto_find_checkpoints
from data_utils import load_traj_csv
from mlp_ik import IKMLP
from gnn_ik import IK_GNN
from trajectory_rollout import build_gnn_input_graph
from classical_ik import (
    connect_pybullet,
    load_kuka,
    get_revolute_joints,
    fk_position,
    get_joint_limits,
)
from torch_geometric.data import Data


def get_device(arg: str) -> torch.device:
    if arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(arg)


def main():
    parser = argparse.ArgumentParser(
        description="Simple inference demo: Given a target pose, predict trajectory."
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="data/kuka_traj_dataset_traj.csv",
        help="Path to trajectory CSV dataset",
    )
    parser.add_argument(
        "--use-orientation",
        action="store_true",
        help="If set, pose is xyz+quat (7D); else xyz (3D).",
    )
    parser.add_argument(
        "--mlp-ckpt",
        type=str,
        default=None,
        help="Path to MLP checkpoint (auto-finds if not provided)",
    )
    parser.add_argument(
        "--gnn-ckpt",
        type=str,
        default=None,
        help="Path to GNN checkpoint (auto-finds if not provided)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=10,
        help="Number of trajectory steps to predict",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: auto, cpu, cuda, mps",
    )
    args = parser.parse_args()

    device = get_device(args.device)

    # Auto-find checkpoints if not provided
    mlp_ckpt = args.mlp_ckpt
    gnn_ckpt = args.gnn_ckpt

    if mlp_ckpt is None or gnn_ckpt is None:
        print("Auto-finding checkpoints...")
        auto_mlp, auto_gnn = auto_find_checkpoints(
            mlp_checkpoint_dir="mlp_ik_traj_checkpoints",
            gnn_checkpoint_dir="gnn_ik_checkpoints",
            model_type="traj",
        )
        if mlp_ckpt is None:
            mlp_ckpt = auto_mlp
            if mlp_ckpt:
                print(f"  Found MLP checkpoint: {mlp_ckpt}")
        if gnn_ckpt is None:
            gnn_ckpt = auto_gnn
            if gnn_ckpt:
                print(f"  Found GNN checkpoint: {gnn_ckpt}")

    if mlp_ckpt is None or gnn_ckpt is None:
        print("Error: Could not find checkpoints. Please train models first.")
        return

    # Load data to get example poses
    print("\nLoading dataset...")
    poses, q_prev, q_curr, pose_dim = load_traj_csv(
        args.csv_path, use_orientation=args.use_orientation
    )
    print(f"Loaded {len(poses)} samples with pose_dim={pose_dim}")

    # Setup PyBullet for FK verification
    print("\nSetting up PyBullet...")
    client_id = connect_pybullet(gui=False)
    kuka_uid = load_kuka()
    joint_indices = get_revolute_joints(kuka_uid)
    ee_link_index = joint_indices[-1]

    # Load models
    print("\nLoading models...")
    mlp = IKMLP.load_from_checkpoint(str(mlp_ckpt), map_location="cpu")
    mlp.to(device)
    mlp.eval()

    gnn = IK_GNN.load_from_checkpoint(
        str(gnn_ckpt),
        node_input_dim=pose_dim + 2,
        output_dim=7,
        map_location="cpu",
    )
    gnn.to(device)
    gnn.eval()

    # Sample a starting pose and initial joint configuration
    rng = np.random.default_rng(42)
    start_idx = rng.integers(0, len(poses))
    target_poses = poses[start_idx : start_idx + args.num_steps]
    q_current_mlp = q_prev[start_idx].copy()
    q_current_gnn = q_prev[start_idx].copy()

    print(f"\n{'='*60}")
    print("INFERENCE DEMO: Predicting Trajectory from Target Poses")
    print(f"{'='*60}")
    print(f"\nStarting from sample {start_idx}")
    print(f"Predicting {args.num_steps} steps")
    print(f"\n{'Step':<6} {'Target Pose (xyz)':<25} {'MLP q[0]':<12} {'GNN q[0]':<12} {'MLP EE Error':<15} {'GNN EE Error':<15}")
    print("-" * 90)

    mlp_trajectory = [q_current_mlp.copy()]
    gnn_trajectory = [q_current_gnn.copy()]

    with torch.no_grad():
        for step, target_pose in enumerate(target_poses):
            target_pos = target_pose[:3]
            target_orn = target_pose[3:7] if pose_dim == 7 else None

            # MLP prediction
            mlp_input = np.concatenate([target_pose, q_current_mlp]).astype(np.float32)
            mlp_input_tensor = torch.from_numpy(mlp_input).unsqueeze(0).to(device)
            mlp_dq = mlp(mlp_input_tensor).cpu().numpy()[0]
            q_current_mlp = q_current_mlp + mlp_dq
            mlp_trajectory.append(q_current_mlp.copy())

            # GNN prediction
            gnn_graph = build_gnn_input_graph(
                target_pose, q_current_gnn, pose_dim, n_joints=7
            )
            gnn_graph = gnn_graph.to(device)
            gnn_dq = gnn(gnn_graph).cpu().numpy()
            q_current_gnn = q_current_gnn + gnn_dq
            gnn_trajectory.append(q_current_gnn.copy())

            # Compute EE errors using FK
            mlp_ee_pos, _ = fk_position(
                kuka_uid, joint_indices, ee_link_index, q_current_mlp
            )
            gnn_ee_pos, _ = fk_position(
                kuka_uid, joint_indices, ee_link_index, q_current_gnn
            )

            mlp_error = np.linalg.norm(mlp_ee_pos - target_pos)
            gnn_error = np.linalg.norm(gnn_ee_pos - target_pos)

            print(
                f"{step+1:<6} "
                f"[{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]  "
                f"{q_current_mlp[0]:>10.4f}  "
                f"{q_current_gnn[0]:>10.4f}  "
                f"{mlp_error:>13.6f}  "
                f"{gnn_error:>13.6f}"
            )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nPredicted {args.num_steps} steps starting from sample {start_idx}")
    print(f"\nFinal joint configurations:")
    print(f"  MLP: {q_current_mlp}")
    print(f"  GNN: {q_current_gnn}")
    print(f"\nFinal EE positions:")
    mlp_final_ee, _ = fk_position(
        kuka_uid, joint_indices, ee_link_index, q_current_mlp
    )
    gnn_final_ee, _ = fk_position(
        kuka_uid, joint_indices, ee_link_index, q_current_gnn
    )
    print(f"  Target: {target_poses[-1][:3]}")
    print(f"  MLP:    {mlp_final_ee}")
    print(f"  GNN:    {gnn_final_ee}")
    print(f"\nFinal EE errors:")
    print(f"  MLP: {np.linalg.norm(mlp_final_ee - target_poses[-1][:3]):.6f} m")
    print(f"  GNN: {np.linalg.norm(gnn_final_ee - target_poses[-1][:3]):.6f} m")

    # Cleanup
    import pybullet as p

    p.disconnect(client_id)
    print("\nDone!")


if __name__ == "__main__":
    main()

