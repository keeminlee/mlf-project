"""
trajectory_rollout.py

Run sequential rollout experiments for IK models on the KUKA iiwa:

- Start from an initial joint configuration q0.
- For a sequence of target end-effector poses {pose_t}, repeatedly apply
  the model with (pose_t, q_t) to get q_{t+1}.
- Use forward kinematics to measure EE position error and joint movement
  over the rollout.

Supports:
  - GNN IK model (Δq-based, required)
  - MLP IK model (optional, assumed to output absolute joint angles)

This is meant as a *qualitative* and *quantitative* stability check beyond
one-step evaluation.

Example usage:

  python src/trajectory_rollout.py \
    --csv-path kuka_traj_dataset_traj.csv \
    --use-orientation \
    --gnn-ckpt gnn_ik_checkpoints/gnnik-epoch=XXX-val_loss=YYY.ckpt \
    --mlp-ckpt mlp_ik_checkpoints/ikmlp-epoch=AAA-val_loss=BBB.ckpt \
    --num-trajectories 10 \
    --traj-length 30
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, List

import argparse
import numpy as np
import torch
from torch_geometric.data import Data

from gnn_ik import IK_GNN
from data_utils import load_traj_csv
from mlp_ik import IKMLP
from checkpoint_utils import auto_find_checkpoints
from classical_ik import (
    connect_pybullet,
    load_kuka,
    get_revolute_joints,
    fk_position,
)


# ------------------------------------------------------------------------------
# Device util
# ------------------------------------------------------------------------------

def get_device(arg: str) -> torch.device:
    if arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(arg)


# ------------------------------------------------------------------------------
# GNN input graph construction
# ------------------------------------------------------------------------------

def build_gnn_input_graph(
    pose: np.ndarray,
    q_prev: np.ndarray,
    pose_dim: int,
    n_joints: int = 7,
) -> Data:
    """
    Build a single PyG Data graph for the GNN IK model, matching the
    KukaTrajGraphDataset convention from gnn_ik.py.

    Node layout:
      - node 0: pose node
      - nodes 1..7: joint nodes

    Node features:
      pose node:
        [pose, 0, 0]
      joint node j:
        [0...0, q_prev_j, j/6]

    Edges:
      - chain edges between joint nodes: (1 <-> 2), ..., (6 <-> 7)
      - pose node (0) connected to each joint node (1..7)
    """
    pose = np.asarray(pose, dtype=np.float32)
    q_prev = np.asarray(q_prev, dtype=np.float32)
    assert pose.shape[0] == pose_dim
    assert q_prev.shape[0] == n_joints

    num_nodes = 1 + n_joints
    node_feat_dim = pose_dim + 2

    # Node features
    x = torch.zeros((num_nodes, node_feat_dim), dtype=torch.float32)

    # Pose node
    x[0, :pose_dim] = torch.from_numpy(pose)

    # Joint nodes
    for j in range(n_joints):
        node_idx = j + 1
        x[node_idx, pose_dim] = float(q_prev[j])
        x[node_idx, pose_dim + 1] = float(j) / float(n_joints - 1)

    # Edges
    edges: List[tuple[int, int]] = []

    # Chain edges between joints (1..7)
    for j in range(1, n_joints):
        a = j
        b = j + 1
        edges.append((a, b))
        edges.append((b, a))

    # Pose node <-> joint nodes
    for j in range(1, n_joints + 1):
        edges.append((0, j))
        edges.append((j, 0))

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # Single-graph batch vector (all nodes belong to graph 0)
    batch = torch.zeros(num_nodes, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, batch=batch)
    # q_prev is convenient to keep around for Δq metrics
    data.q_prev = torch.from_numpy(q_prev.astype(np.float32))
    return data


# ------------------------------------------------------------------------------
# Rollout core
# ------------------------------------------------------------------------------

def rollout_model(
    model_type: str,
    model: torch.nn.Module,
    poses: np.ndarray,
    q_init: np.ndarray,
    pose_dim: int,
    device: torch.device,
    kuka_uid: int,
    joint_indices,
    ee_link_index: int,
) -> Dict[str, Any]:
    """
    Roll out a single trajectory for either the MLP or GNN model.

    Args:
        model_type:  "mlp" or "gnn"
        model:       IKMLP or IK_GNN instance
        poses:       (T, pose_dim) sequence of desired poses
        q_init:      (7,) initial joint configuration
        pose_dim:    3 or 7
        device:      torch.device
        kuka_uid:    PyBullet KUKA body id
        joint_indices: list of joint indices
        ee_link_index: end-effector link index

    Returns:
        dict with lists:
          - "ee_errors":    [T] EE position error (L2) per step
          - "dq_norms":     [T] ||Δq||_2 per step
          - "dq_norms_L1":  [T] ||Δq||_1 per step
    """
    T = poses.shape[0]
    q = np.asarray(q_init, dtype=np.float32).copy()
    assert q.shape[0] == 7

    ee_errors: List[float] = []
    dq_norms: List[float] = []
    dq_norms_L1: List[float] = []

    model.eval()

    with torch.no_grad():
        for t in range(T):
            pose_t = poses[t]  # (pose_dim,)

            if model_type == "gnn":
                # Build graph input, feed through GNN, interpret output as Δq
                data = build_gnn_input_graph(pose_t, q, pose_dim, n_joints=7)
                data = data.to(device)
                dq_pred = model(data)  # (1, 7)
                dq = dq_pred[0].cpu().numpy().astype(np.float32)
                q_new = q + dq

            elif model_type == "mlp":
                # Determine whether MLP is Δq-based or absolute
                predict_delta = bool(getattr(model.hparams, "predict_delta", False))

                input_dim = int(getattr(model.hparams, "input_dim", pose_dim))
                if input_dim == pose_dim + 7:
                    # trajectory MLP input = [pose, q_prev]
                    x = np.concatenate([pose_t, q], axis=0).astype(np.float32)
                elif input_dim == pose_dim:
                    # single-shot (absolute q) model
                    x = pose_t.astype(np.float32)
                else:
                    raise ValueError(
                        f"Unexpected MLP input_dim={input_dim}, pose_dim={pose_dim}."
                    )

                x_t = torch.from_numpy(x[None, :]).to(device)
                out = model(x_t)[0].cpu().numpy().astype(np.float32)

                if predict_delta:
                    # Δq model: behaves like GNN
                    dq = out
                    q_new = q + dq
                else:
                    # absolute-q model
                    dq = out - q
                    q_new = out
            else:
                raise ValueError(f"Unknown model_type: {model_type}")

            # FK to measure EE error vs target pose
            ee_pos = fk_position(kuka_uid, joint_indices, ee_link_index, q_new)
            target_pos = pose_t[:3]  # always xyz at the start of pose vector
            ee_err = float(np.linalg.norm(ee_pos - target_pos))

            ee_errors.append(ee_err)
            dq_norms.append(float(np.linalg.norm(dq, ord=2)))
            dq_norms_L1.append(float(np.sum(np.abs(dq))))

            q = q_new

    return {
        "ee_errors": ee_errors,
        "dq_norms": dq_norms,
        "dq_norms_L1": dq_norms_L1,
    }


# ------------------------------------------------------------------------------
# CLI + main
# ------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Sequential rollout experiment for MLP and GNN IK models."
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
        help="If set, pose is xyz+quat (7D); else xyz (3D).",
    )
    ap.add_argument(
        "--mlp-ckpt",
        type=str,
        default=None,
        help="Path to MLP checkpoint (.ckpt). If omitted, MLP rollout is skipped.",
    )
    ap.add_argument(
        "--gnn-ckpt",
        type=str,
        default=None,
        help="Path to GNN checkpoint (.ckpt). Auto-finds if not provided.",
    )
    ap.add_argument(
        "--num-trajectories",
        type=int,
        default=10,
        help="Number of random rollout trajectories to simulate.",
    )
    ap.add_argument(
        "--traj-length",
        type=int,
        default=30,
        help="Number of steps per trajectory.",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'auto', 'cpu', 'cuda', or 'mps'.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(args.device)

    # Auto-find checkpoints if not provided
    mlp_ckpt_path = args.mlp_ckpt
    gnn_ckpt_path = args.gnn_ckpt
    
    if mlp_ckpt_path is None or gnn_ckpt_path is None:
        print("Auto-finding checkpoints...")
        auto_mlp, auto_gnn = auto_find_checkpoints(
            mlp_checkpoint_dir="mlp_ik_traj_checkpoints",
            gnn_checkpoint_dir="gnn_ik_checkpoints",
            model_type="traj",
        )
        if mlp_ckpt_path is None:
            mlp_ckpt_path = auto_mlp
            if mlp_ckpt_path:
                print(f"  Found MLP checkpoint: {mlp_ckpt_path}")
        if gnn_ckpt_path is None:
            gnn_ckpt_path = auto_gnn
            if gnn_ckpt_path:
                print(f"  Found GNN checkpoint: {gnn_ckpt_path}")
    
    if gnn_ckpt_path is None:
        raise ValueError(
            "GNN checkpoint required but not found! Please either:\n"
            "  1. Provide checkpoint path with --gnn-ckpt\n"
            "  2. Train GNN model first (see README Section 3)\n"
            "  3. Check that gnn_ik_checkpoints/ directory exists"
        )

    # Load trajectory dataset (poses, q_prev, q_curr)
    poses, q_prev, q_curr, pose_dim = load_traj_csv(
        args.csv_path,
        use_orientation=args.use_orientation,
    )
    N = poses.shape[0]
    n_joints = q_prev.shape[1]
    assert n_joints == 7

    # PyBullet setup for FK
    client_id = connect_pybullet(gui=False)
    kuka_uid = load_kuka()
    joint_indices = get_revolute_joints(kuka_uid)
    ee_link_index = joint_indices[-1]

    # Models
    # ------
    # GNN
    gnn_ckpt = Path(gnn_ckpt_path)
    # Need node_input_dim consistent with training: pose_dim + 2
    node_input_dim = pose_dim + 2
    gnn: IK_GNN = IK_GNN.load_from_checkpoint(
        gnn_ckpt,
        node_input_dim=node_input_dim,
        output_dim=n_joints,
    )
    gnn.to(device)

    # MLP (optional)
    mlp: Optional[IKMLP] = None
    if mlp_ckpt_path is not None:
        mlp_ckpt = Path(mlp_ckpt_path)
        mlp = IKMLP.load_from_checkpoint(mlp_ckpt)
        mlp.to(device)

    # Rollout experiment
    rng = np.random.default_rng(0)

    def sample_indices():
        # For each trajectory, sample traj_length target poses (with replacement).
        return rng.integers(low=0, high=N, size=args.traj_length, endpoint=False)

    results = {
        "gnn": {"ee_errors": [], "dq_norms": [], "dq_norms_L1": []},
        "mlp": {"ee_errors": [], "dq_norms": [], "dq_norms_L1": []},
    }

    for traj_idx in range(args.num_trajectories):
        idxs = sample_indices()
        poses_traj = poses[idxs]         # (T, pose_dim)
        q_init = q_prev[idxs[0]]         # start from a dataset q_prev

        # GNN rollout
        gnn_res = rollout_model(
            model_type="gnn",
            model=gnn,
            poses=poses_traj,
            q_init=q_init,
            pose_dim=pose_dim,
            device=device,
            kuka_uid=kuka_uid,
            joint_indices=joint_indices,
            ee_link_index=ee_link_index,
        )
        results["gnn"]["ee_errors"].extend(gnn_res["ee_errors"])
        results["gnn"]["dq_norms"].extend(gnn_res["dq_norms"])
        results["gnn"]["dq_norms_L1"].extend(gnn_res["dq_norms_L1"])

        # MLP rollout (if provided)
        if mlp is not None:
            mlp_res = rollout_model(
                model_type="mlp",
                model=mlp,
                poses=poses_traj,
                q_init=q_init,
                pose_dim=pose_dim,
                device=device,
                kuka_uid=kuka_uid,
                joint_indices=joint_indices,
                ee_link_index=ee_link_index,
            )
            results["mlp"]["ee_errors"].extend(mlp_res["ee_errors"])
            results["mlp"]["dq_norms"].extend(mlp_res["dq_norms"])
            results["mlp"]["dq_norms_L1"].extend(mlp_res["dq_norms_L1"])

    # Aggregate and print summary
    def summarize(name: str):
        if len(results[name]["ee_errors"]) == 0:
            return
        ee_err = np.array(results[name]["ee_errors"], dtype=np.float32)
        dq_norm = np.array(results[name]["dq_norms"], dtype=np.float32)
        dq_norm1 = np.array(results[name]["dq_norms_L1"], dtype=np.float32)

        print(f"\n=== {name.upper()} rollout summary over "
              f"{args.num_trajectories} trajectories (T={args.traj_length}) ===")
        print(f"{name.upper()} mean EE error (L2):  {ee_err.mean():.6f}")
        print(f"{name.upper()} std  EE error (L2):  {ee_err.std():.6f}")
        print(f"{name.upper()} mean ||Δq||_2:      {dq_norm.mean():.6f}")
        print(f"{name.upper()} mean ||Δq||_1:      {dq_norm1.mean():.6f}")

    summarize("gnn")
    if mlp is not None:
        summarize("mlp")

    # Clean up PyBullet
    import pybullet as p
    p.disconnect(client_id)


if __name__ == "__main__":
    main()
