# kuka_fk_dataset.py
import math
import random
import argparse
from pathlib import Path
from typing import List

import numpy as np
import pybullet as p
import pybullet_data as pd

from classical_ik import get_revolute_joints, get_joint_limits


def sample_joints(lowers: np.ndarray, uppers: np.ndarray, rng: random.Random) -> np.ndarray:
    """Uniform sample within joint limits."""
    return np.array([rng.uniform(lo, hi) for lo, hi in zip(lowers, uppers)], dtype=np.float32)


def generate_trajectory_data(
    kuka_uid: int,
    joint_indices: List[int],
    joint_lowers: np.ndarray,
    joint_uppers: np.ndarray,
    ee_link_index: int,
    num_trajectories: int,
    steps_per_trajectory: int,
    step_std: float,
    include_orientation: bool,
    rng: np.random.Generator,
    csv_path: str | Path,
) -> None:
    """
    Generate trajectory-style IK dataset.

    Each row encodes:
        [pose_t, q_prev (t-1), q_curr (t)]

    where:
        - pose_t is end-effector pose at time t (xyz or xyz+quat)
        - q_prev is previous joint configuration
        - q_curr is current joint configuration that produced pose_t

    Args:
        kuka_uid:            PyBullet body unique ID for KUKA iiwa
        joint_indices:       list of movable joint indices (7 for KUKA iiwa)
        joint_lowers:        (n_joints,) array of lower joint limits
        joint_uppers:        (n_joints,) array of upper joint limits
        ee_link_index:       link index of the end-effector
        num_trajectories:    number of independent joint-space trajectories
        steps_per_trajectory:number of steps per trajectory
        step_std:            std-dev of joint-space random walk (radians)
        include_orientation: if True, include quaternion in pose (7D pose),
                             otherwise only xyz (3D pose)
        rng:                 np.random.Generator for reproducibility
        csv_path:            output CSV path
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    n_joints = len(joint_indices)
    rows = []

    # Build header
    pose_cols = ["ee_x", "ee_y", "ee_z"]
    if include_orientation:
        pose_cols += ["ee_qx", "ee_qy", "ee_qz", "ee_qw"]

    q_prev_cols = [f"q_prev_{j}" for j in range(n_joints)]
    q_curr_cols = [f"q_curr_{j}" for j in range(n_joints)]
    header = pose_cols + q_prev_cols + q_curr_cols

    for traj_idx in range(num_trajectories):
        # Sample a random starting configuration for this trajectory
        q_prev = rng.uniform(low=joint_lowers, high=joint_uppers, size=n_joints)

        for step in range(steps_per_trajectory):
            # Sample a small joint-space step
            dq = rng.normal(loc=0.0, scale=step_std, size=n_joints)
            q_curr = q_prev + dq
            # Clamp to joint limits
            q_curr = np.clip(q_curr, joint_lowers, joint_uppers)

            # Set robot to q_curr and compute FK
            for j_idx, qj in zip(joint_indices, q_curr):
                p.resetJointState(kuka_uid, j_idx, float(qj))

            link_state = p.getLinkState(
                kuka_uid,
                ee_link_index,
                computeForwardKinematics=True,
            )
            pos = link_state[4]  # worldLinkFramePosition
            orn = link_state[5]  # worldLinkFrameOrientation (quaternion)

            if include_orientation:
                pose = list(pos) + list(orn)
            else:
                pose = list(pos)

            row = pose + q_prev.tolist() + q_curr.tolist()
            rows.append(row)

            # Next step uses current config as previous
            q_prev = q_curr

    rows = np.asarray(rows, dtype=np.float32)

    # Save as CSV with header
    np.savetxt(
        csv_path,
        rows,
        delimiter=",",
        fmt="%.6f",
        header=",".join(header),
        comments="",  # no leading '#' on header line
    )
    print(f"[generate_trajectory_data] Saved {len(rows)} samples to {csv_path}")


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Generate IK training data for KUKA iiwa.\n\n"
            "Two modes:\n"
            "  - fk:   i.i.d. FK samples (EE pose -> joint angles)\n"
            "  - traj: trajectory samples (pose_t, q_prev, q_curr)\n"
        )
    )
    # Common args
    ap.add_argument(
        "--data-type",
        type=str,
        choices=["fk", "traj"],
        default="fk",
        help=(
            "Type of dataset to generate:\n"
            "  'fk'   → one row per independent FK sample (EE pose, joint_0..6).\n"
            "  'traj' → one row per trajectory step (pose_t, q_prev_*, q_curr_*)."
        ),
    )
    ap.add_argument(
        "--num-samples",
        type=int,
        default=5000,
        help="Number of FK samples to generate (only used in 'fk' mode).",
    )
    ap.add_argument("--seed", type=int, default=42, help="RNG seed")
    ap.add_argument(
        "--include-orientation",
        action="store_true",
        help="If set, include EE orientation (quat) in pose in addition to xyz.",
    )
    ap.add_argument(
        "--save-npz",
        action="store_true",
        help="Also save a compressed .npz with X,y arrays (fk mode only).",
    )
    ap.add_argument(
        "--out-prefix",
        type=str,
        default="kuka_fk_dataset",
        help="Output file prefix (CSV and optional NPZ).",
    )
    ap.add_argument(
        "--gravity",
        type=float,
        default=-9.81,
        help="Gravity z (not critical for FK with resets).",
    )

    # Trajectory-specific args
    ap.add_argument(
        "--num-trajectories",
        type=int,
        default=200,
        help="Number of joint-space trajectories to generate (traj mode).",
    )
    ap.add_argument(
        "--steps-per-trajectory",
        type=int,
        default=50,
        help="Number of steps per trajectory (traj mode).",
    )
    ap.add_argument(
        "--traj-step-std",
        type=float,
        default=0.05,
        help="Std-dev of joint-space random walk per step in radians (traj mode).",
    )

    args = ap.parse_args()

    # RNGs: Python for FK sampling, NumPy for trajectories
    rng_py = random.Random(args.seed)
    rng_np = np.random.default_rng(args.seed)

    # Headless physics
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pd.getDataPath())
    p.setGravity(0, 0, args.gravity)

    # Load a plane (not strictly needed) and KUKA iiwa
    p.loadURDF("plane.urdf")
    kuka_uid = p.loadURDF(
        "kuka_iiwa/model.urdf",
        useFixedBase=True,
        flags=p.URDF_USE_INERTIA_FROM_FILE,
    )

    # Identify movable joints and limits
    joint_indices = get_revolute_joints(kuka_uid)
    if len(joint_indices) == 0:
        raise RuntimeError("No movable joints detected. Check the URDF path.")
    joint_l, joint_u = get_joint_limits(kuka_uid, joint_indices)

    # End-effector link index: use the last movable joint's link
    ee_link_index = joint_indices[-1]

    # ------------------------------------------------------------------
    # TRAJECTORY MODE: (pose_t, q_prev, q_curr)
    # ------------------------------------------------------------------
    if args.data_type == "traj":
        csv_path = f"{args.out_prefix}_traj.csv"

        generate_trajectory_data(
            kuka_uid=kuka_uid,
            joint_indices=joint_indices,
            joint_lowers=joint_l,
            joint_uppers=joint_u,
            ee_link_index=ee_link_index,
            num_trajectories=args.num_trajectories,
            steps_per_trajectory=args.steps_per_trajectory,
            step_std=args.traj_step_std,
            include_orientation=args.include_orientation,
            rng=rng_np,
            csv_path=csv_path,
        )

        print(
            f"Trajectory dataset saved to {csv_path} "
            f"(~{args.num_trajectories * args.steps_per_trajectory} rows)."
        )
        p.disconnect()
        return

    # ------------------------------------------------------------------
    # FK MODE: i.i.d. (pose, joints) pairs
    # ------------------------------------------------------------------
    X_list = []  # inputs: EE pose
    y_list = []  # targets: joint angles

    for _ in range(args.num_samples):
        q = sample_joints(joint_l, joint_u, rng_py)

        # Reset all joints
        for j, qj in zip(joint_indices, q):
            p.resetJointState(kuka_uid, j, float(qj))

        # Forward kinematics via link state
        ls = p.getLinkState(kuka_uid, ee_link_index, computeForwardKinematics=True)
        pos = ls[4]  # worldLinkFramePosition (x,y,z)
        orn = ls[5]  # worldLinkFrameOrientation (x,y,z,w) quaternion

        if args.include_orientation:
            x = list(pos) + list(orn)  # 7-dim input
        else:
            x = list(pos)  # 3-dim input

        X_list.append(x)
        y_list.append(q.tolist())

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    # Save CSV (columns: ee pose then joints)
    if args.include_orientation:
        ee_cols = ["ee_x", "ee_y", "ee_z", "ee_qx", "ee_qy", "ee_qz", "ee_qw"]
    else:
        ee_cols = ["ee_x", "ee_y", "ee_z"]
    joint_cols = [f"joint_{i}" for i in range(len(joint_indices))]

    header = ee_cols + joint_cols
    csv_path = f"{args.out_prefix}.csv"

    xy = np.concatenate([X, y], axis=1)
    np.savetxt(
        csv_path,
        xy,
        delimiter=",",
        header=",".join(header),
        comments="",
    )
    print(f"Saved CSV: {csv_path}  (shape: X={X.shape}, y={y.shape})")

    if args.save_npz:
        npz_path = f"{args.out_prefix}.npz"
        np.savez_compressed(
            npz_path,
            X=X,
            y=y,
            joint_indices=np.array(joint_indices, dtype=np.int32),
        )
        print(f"Saved NPZ: {npz_path}")

    p.disconnect()


if __name__ == "__main__":
    main()
