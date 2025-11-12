# kuka_fk_dataset.py
import os
import math
import random
import argparse
from typing import List, Tuple

import numpy as np
import pybullet as p
import pybullet_data as pd


def get_revolute_joints(body_id: int) -> List[int]:
    """Return indices of all revolute/prismatic joints in order."""
    n_joints = p.getNumJoints(body_id)
    movable = []
    for j in range(n_joints):
        ji = p.getJointInfo(body_id, j)
        joint_type = ji[2]
        # 0=REVOLUTE, 1=PRISMATIC, 4=GEAR (skip), others fixed
        if joint_type in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
            movable.append(j)
    return movable


def get_joint_limits(body_id: int, joint_indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Read lower/upper limits from URDF; fall back to [-pi, pi] if invalid."""
    lowers, uppers = [], []
    for j in joint_indices:
        ji = p.getJointInfo(body_id, j)
        lo, hi = ji[8], ji[9]  # jointLowerLimit, jointUpperLimit
        if hi < lo or abs(lo) == 0.0 and abs(hi) == 0.0:
            lo, hi = -math.pi, math.pi
        lowers.append(lo)
        uppers.append(hi)
    return np.array(lowers, dtype=np.float32), np.array(uppers, dtype=np.float32)


def sample_joints(lowers: np.ndarray, uppers: np.ndarray, rng: random.Random) -> np.ndarray:
    """Uniform sample within joint limits."""
    return np.array([rng.uniform(lo, hi) for lo, hi in zip(lowers, uppers)], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser(description="Generate FK pairs for KUKA iiwa: (EE pose -> joint angles).")
    ap.add_argument("--num-samples", type=int, default=5000, help="Number of FK samples to generate")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed")
    ap.add_argument("--include-orientation", action="store_true",
                    help="If set, include ee orientation (quat) in inputs in addition to xyz")
    ap.add_argument("--save-npz", action="store_true", help="Also save a compressed .npz with X,y arrays")
    ap.add_argument("--out-prefix", type=str, default="kuka_fk_dataset",
                    help="Output file prefix (CSV and optional NPZ)")
    ap.add_argument("--gravity", type=float, default=-9.81, help="Gravity z (not critical for FK with resets)")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # Headless physics
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pd.getDataPath())
    p.setGravity(0, 0, args.gravity)

    # Load a plane (not strictly needed) and KUKA iiwa
    p.loadURDF("plane.urdf")
    # Common path in pybullet_data: "kuka_iiwa/model.urdf"
    kuka_uid = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True, flags=p.URDF_USE_INERTIA_FROM_FILE)

    # Identify movable joints and limits
    joint_indices = get_revolute_joints(kuka_uid)
    if len(joint_indices) == 0:
        raise RuntimeError("No movable joints detected. Check the URDF path.")
    joint_l, joint_u = get_joint_limits(kuka_uid, joint_indices)

    # End-effector link index: use the last movable joint's link
    ee_link_index = joint_indices[-1]

    # Buffers
    X_list = []  # inputs: ee pose
    y_list = []  # targets: joint angles

    # Generate samples
    for _ in range(args.num_samples):
        q = sample_joints(joint_l, joint_u, rng)

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
            x = list(pos)              # 3-dim input

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
    # Stack X and y horizontally for CSV
    xy = np.concatenate([X, y], axis=1)
    np.savetxt(csv_path, xy, delimiter=",", header=",".join(header), comments="")
    print(f"Saved CSV: {csv_path}  (shape: X={X.shape}, y={y.shape})")

    if args.save_npz:
        npz_path = f"{args.out_prefix}.npz"
        np.savez_compressed(npz_path, X=X, y=y, joint_indices=np.array(joint_indices, dtype=np.int32))
        print(f"Saved NPZ: {npz_path}")

    p.disconnect()


if __name__ == "__main__":
    main()
