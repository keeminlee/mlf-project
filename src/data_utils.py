"""
data_utils.py

Shared data loading utilities for IK models.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
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
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)

    pose_dim = 7 if use_orientation else 3
    num_cols = data.shape[1]

    expected_cols = pose_dim + 14  # pose + 7 q_prev + 7 q_curr
    if num_cols != expected_cols:
        raise ValueError(
            f"Expected TRAJ CSV with {expected_cols} columns "
            f"(pose_dim={pose_dim} + 14), got {num_cols}."
        )

    poses = data[:, :pose_dim]
    q_prev = data[:, pose_dim:pose_dim + 7]
    q_curr = data[:, pose_dim + 7:]

    return (
        poses.astype(np.float32),
        q_prev.astype(np.float32),
        q_curr.astype(np.float32),
        pose_dim,
    )

