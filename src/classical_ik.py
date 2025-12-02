"""
classical_ik.py

Classical forward and inverse kinematics utilities for the 7-DoF KUKA iiwa
arm in PyBullet.

Provides:

- connect_pybullet(gui=False): connect to PyBullet and set basic physics params
- load_kuka():                 load kuka_iiwa/model.urdf with fixed base
- get_revolute_joints():       return list of revolute joint indices
- get_joint_limits():          return joint lower/upper bounds
- fk_position():               forward kinematics (xyz)
- fk_pose():                   forward kinematics (xyz + quaternion)
- compute_jacobian():          linear & angular Jacobians via PyBullet
- ik_damped_least_squares():   numerical IK (DLS) using FK + Jacobian
- ik_pybullet_builtin():       wrapper around PyBullet's built-in IK
"""

from __future__ import annotations

from typing import List, Tuple, Optional, Dict

import numpy as np
import pybullet as p
import pybullet_data as pd


# ------------------------------------------------------------------------------
# Basic setup
# ------------------------------------------------------------------------------

def connect_pybullet(gui: bool = False, gravity: float = -9.81) -> int:
    """
    Connect to PyBullet and set gravity & search path.

    Args:
        gui:     if True, use GUI mode; otherwise DIRECT (headless)
        gravity: z component of gravity

    Returns:
        client_id: PyBullet client id
    """
    if gui:
        cid = p.connect(p.GUI)
    else:
        cid = p.connect(p.DIRECT)

    p.setAdditionalSearchPath(pd.getDataPath())
    p.setGravity(0, 0, gravity)
    # Optional plane (not strictly needed for FK/IK)
    p.loadURDF("plane.urdf")
    return cid


def load_kuka() -> int:
    """
    Load the KUKA iiwa URDF with fixed base.

    Returns:
        kuka_uid: PyBullet body unique id
    """
    kuka_uid = p.loadURDF(
        "kuka_iiwa/model.urdf",
        useFixedBase=True,
        flags=p.URDF_USE_INERTIA_FROM_FILE,
    )
    return kuka_uid


def get_revolute_joints(body_uid: int) -> List[int]:
    """
    Get indices of all revolute joints for the given body.

    For KUKA iiwa, this should return 7 joints.

    Args:
        body_uid: PyBullet body unique id

    Returns:
        list of joint indices (int)
    """
    num_joints = p.getNumJoints(body_uid)
    revolute = []
    for j in range(num_joints):
        info = p.getJointInfo(body_uid, j)
        joint_type = info[2]
        if joint_type == p.JOINT_REVOLUTE:
            revolute.append(j)
    return revolute


def get_joint_limits(
    body_uid: int,
    joint_indices: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get joint lower/upper limits for a subset of joints.
    Falls back to [-pi, pi] if limits are invalid.

    Args:
        body_uid:       PyBullet body unique id
        joint_indices:  list of joint indices

    Returns:
        joint_lowers: (n_joints,) array
        joint_uppers: (n_joints,) array
    """
    import math
    lowers = []
    uppers = []
    for j in joint_indices:
        info = p.getJointInfo(body_uid, j)
        lo, hi = info[8], info[9]  # joint lower limit, joint upper limit
        # Fall back to [-pi, pi] if limits are invalid
        if hi < lo or (abs(lo) == 0.0 and abs(hi) == 0.0):
            lo, hi = -math.pi, math.pi
        lowers.append(lo)
        uppers.append(hi)
    return np.array(lowers, dtype=np.float32), np.array(uppers, dtype=np.float32)


# ------------------------------------------------------------------------------
# Forward kinematics
# ------------------------------------------------------------------------------

def fk_pose(
    body_uid: int,
    joint_indices: List[int],
    ee_link_index: int,
    q: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Forward kinematics: map joint angles to end-effector pose (position + quaternion).

    Args:
        body_uid:       PyBullet body unique id
        joint_indices:  list of joint indices (length n_joints)
        ee_link_index:  link index of the end-effector
        q:              (n_joints,) joint angles (radians)

    Returns:
        pos: (3,) EE position (world frame)
        orn: (4,) EE orientation quaternion (x, y, z, w)
    """
    q = np.asarray(q, dtype=np.float32)
    assert q.shape[0] == len(joint_indices)

    # Reset joint states to q
    for j, qj in zip(joint_indices, q):
        p.resetJointState(body_uid, j, float(qj))

    link_state = p.getLinkState(
        body_uid,
        ee_link_index,
        computeForwardKinematics=True,
    )
    pos = np.array(link_state[4], dtype=np.float32)  # worldLinkFramePosition
    orn = np.array(link_state[5], dtype=np.float32)  # worldLinkFrameOrientation (quat)
    return pos, orn


def fk_position(
    body_uid: int,
    joint_indices: List[int],
    ee_link_index: int,
    q: np.ndarray,
) -> np.ndarray:
    """
    Convenience wrapper for FK position only.

    Args:
        body_uid:       PyBullet body unique id
        joint_indices:  list of joint indices
        ee_link_index:  EE link index
        q:              (n_joints,) joint angles

    Returns:
        pos: (3,) EE position
    """
    pos, _ = fk_pose(body_uid, joint_indices, ee_link_index, q)
    return pos


# ------------------------------------------------------------------------------
# Jacobian via PyBullet
# ------------------------------------------------------------------------------

def compute_jacobian(
    body_uid: int,
    joint_indices: List[int],
    ee_link_index: int,
    q: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the Jacobian of the end-effector wrt joint angles using PyBullet.

    Args:
        body_uid:       PyBullet body unique id
        joint_indices:  list of joint indices
        ee_link_index:  EE link index
        q:              (n_joints,) joint angles

    Returns:
        J_pos: (3, n_joints) linear Jacobian (dpos/dq)
        J_rot: (3, n_joints) angular Jacobian (domega/dq)
    """
    q = np.asarray(q, dtype=np.float32)
    n_joints = len(joint_indices)
    assert q.shape[0] == n_joints

    # PyBullet expects vectors for *all* joints, but we only care about joint_indices.
    # For the KUKA iiwa, it's safe to pass q in the order of joint_indices and zeros elsewhere.
    # Simplest: map them into a full q vector of size getNumJoints, but for the iiwa the
    # revolute joints are usually consecutive, so we can directly use q in that range.
    # To keep it robust, we build full lists:
    num_joints_total = p.getNumJoints(body_uid)
    q_full = [0.0] * num_joints_total
    q_dot_full = [0.0] * num_joints_total
    q_acc_full = [0.0] * num_joints_total

    for idx_local, j in enumerate(joint_indices):
        q_full[j] = float(q[idx_local])

    # Compute Jacobian
    zero_vec = [0.0, 0.0, 0.0]
    J_lin, J_ang = p.calculateJacobian(
        bodyUniqueId=body_uid,
        linkIndex=ee_link_index,
        localPosition=zero_vec,
        objPositions=q_full,
        objVelocities=q_dot_full,
        objAccelerations=q_acc_full,
    )

    J_pos = np.array(J_lin, dtype=np.float32)  # (3, num_joints_total)
    J_rot = np.array(J_ang, dtype=np.float32)

    # Extract columns for our joint_indices
    cols = np.array(joint_indices, dtype=int)
    J_pos = J_pos[:, cols]
    J_rot = J_rot[:, cols]
    return J_pos, J_rot


# ------------------------------------------------------------------------------
# Damped Least-Squares IK
# ------------------------------------------------------------------------------

def ik_damped_least_squares(
    body_uid: int,
    joint_indices: List[int],
    ee_link_index: int,
    target_pos: np.ndarray,
    target_orn: Optional[np.ndarray] = None,
    q_init: Optional[np.ndarray] = None,
    joint_lowers: Optional[np.ndarray] = None,
    joint_uppers: Optional[np.ndarray] = None,
    max_iters: int = 100,
    pos_tol: float = 1e-3,
    orn_tol: float = 1e-2,
    damping: float = 1e-2,
    step_size: float = 1.0,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Damped least-squares IK solver.

    Solves for q that brings the EE close to (target_pos, target_orn) starting
    from q_init. If target_orn is None, only position is matched.

    Args:
        body_uid:      PyBullet body id
        joint_indices: list of joint indices
        ee_link_index: EE link index
        target_pos:    (3,) desired EE position
        target_orn:    (4,) desired EE quaternion, or None to ignore orientation
        q_init:        (n_joints,) initial guess; default zeros
        joint_lowers:  (n_joints,) joint lower limits; if None, no clamping
        joint_uppers:  (n_joints,) joint upper limits
        max_iters:     max number of iterations
        pos_tol:       position error tolerance (L2)
        orn_tol:       orientation error tolerance (L2 of axis-angle approx)
        damping:       DLS damping coefficient λ
        step_size:     step size for updates

    Returns:
        q:         (n_joints,) IK solution (or last iterate)
        info:      dict with keys:
                     'iters', 'pos_err', 'orn_err'
    """
    target_pos = np.asarray(target_pos, dtype=np.float32)
    if target_orn is not None:
        target_orn = np.asarray(target_orn, dtype=np.float32)

    n_joints = len(joint_indices)
    if q_init is None:
        q = np.zeros(n_joints, dtype=np.float32)
    else:
        q = np.asarray(q_init, dtype=np.float32).copy()

    if joint_lowers is not None:
        joint_lowers = np.asarray(joint_lowers, dtype=np.float32)
    if joint_uppers is not None:
        joint_uppers = np.asarray(joint_uppers, dtype=np.float32)

    def clamp(q_vec: np.ndarray) -> np.ndarray:
        if joint_lowers is None or joint_uppers is None:
            return q_vec
        return np.clip(q_vec, joint_lowers, joint_uppers)

    pos_err = np.inf
    orn_err = 0.0

    for it in range(max_iters):
        pos, orn = fk_pose(body_uid, joint_indices, ee_link_index, q)

        # Position error
        e_pos = target_pos - pos
        pos_err = float(np.linalg.norm(e_pos, ord=2))

        # Orientation error as a small 3D axis-angle approx
        if target_orn is not None:
            # Compute relative orientation: target_orn * conj(orn)
            # PyBullet provides utilities, but we'll approximate via p.getDifferenceQuaternion
            rel = p.getDifferenceQuaternion(target_orn.tolist(), orn.tolist())
            # quaternion difference -> axis-angle approx: axis * angle
            # For small angles, sin(theta/2) ~ theta/2, and rel[0:3] ~ axis*sin(theta/2)
            axis = np.array(rel[0:3], dtype=np.float32)
            orn_err = float(2.0 * np.linalg.norm(axis, ord=2))
        else:
            orn_err = 0.0

        if pos_err < pos_tol and (target_orn is None or orn_err < orn_tol):
            break

        J_pos, J_rot = compute_jacobian(body_uid, joint_indices, ee_link_index, q)

        if target_orn is not None:
            # 6D task: [position; orientation]
            e = np.concatenate([e_pos, -axis], axis=0)  # orientation sign doesn't matter much
            J = np.concatenate([J_pos, J_rot], axis=0)  # (6, n_joints)
        else:
            # 3D task: position only
            e = e_pos
            J = J_pos  # (3, n_joints)

        # Damped least squares: dq = J^T (J J^T + λ^2 I)^-1 e
        JT = J.T
        JJt = J @ JT
        lam2I = (damping ** 2) * np.eye(JJt.shape[0], dtype=np.float32)
        dq = JT @ np.linalg.solve(JJt + lam2I, e)

        q = q + step_size * dq.astype(np.float32)
        q = clamp(q)

    info = {
        "iters": float(it + 1),
        "pos_err": pos_err,
        "orn_err": orn_err,
    }
    return q, info


# ------------------------------------------------------------------------------
# Wrapper around PyBullet's built-in IK
# ------------------------------------------------------------------------------

def ik_pybullet_builtin(
    body_uid: int,
    joint_indices: List[int],
    ee_link_index: int,
    target_pos: np.ndarray,
    target_orn: Optional[np.ndarray] = None,
    joint_lowers: Optional[np.ndarray] = None,
    joint_uppers: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Convenience wrapper around PyBullet's built-in IK solver.

    Args:
        body_uid:      PyBullet body id
        joint_indices: list of joint indices
        ee_link_index: EE link index
        target_pos:    (3,) target EE position
        target_orn:    (4,) target EE quat or None (position-only IK)
        joint_lowers:  (n_joints,) joint lower limits (for joint limits variant)
        joint_uppers:  (n_joints,) joint upper limits

    Returns:
        q: (n_joints,) joint angles from PyBullet IK
    """
    target_pos = np.asarray(target_pos, dtype=np.float32)
    if target_orn is not None:
        target_orn = np.asarray(target_orn, dtype=np.float32)

    if joint_lowers is not None and joint_uppers is not None:
        joint_lowers = np.asarray(joint_lowers, dtype=np.float32)
        joint_uppers = np.asarray(joint_uppers, dtype=np.float32)
        # Use joint-limit aware IK
        if target_orn is not None:
            q_full = p.calculateInverseKinematics(
                body_uid,
                ee_link_index,
                target_pos.tolist(),
                target_orn.tolist(),
                lowerLimits=joint_lowers.tolist(),
                upperLimits=joint_uppers.tolist(),
            )
        else:
            q_full = p.calculateInverseKinematics(
                body_uid,
                ee_link_index,
                target_pos.tolist(),
                lowerLimits=joint_lowers.tolist(),
                upperLimits=joint_uppers.tolist(),
            )
    else:
        # Default IK
        if target_orn is not None:
            q_full = p.calculateInverseKinematics(
                body_uid,
                ee_link_index,
                target_pos.tolist(),
                target_orn.tolist(),
            )
        else:
            q_full = p.calculateInverseKinematics(
                body_uid,
                ee_link_index,
                target_pos.tolist(),
            )

    q_full = np.array(q_full, dtype=np.float32)
    # Extract only the joints we care about
    q = q_full[joint_indices]
    return q


# ------------------------------------------------------------------------------
# Simple manual test (optional)
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    # Quick sanity check: solve IK for a random target reachable by the arm.
    cid = connect_pybullet(gui=False)
    kuka_uid = load_kuka()
    joint_indices = get_revolute_joints(kuka_uid)
    joint_l, joint_u = get_joint_limits(kuka_uid, joint_indices)
    ee_link_index = joint_indices[-1]

    # Pick a random configuration, compute its pose, then try to recover q via IK
    q_true = (joint_l + joint_u) / 2.0
    pos_true, orn_true = fk_pose(kuka_uid, joint_indices, ee_link_index, q_true)

    q_ik, info = ik_damped_least_squares(
        body_uid=kuka_uid,
        joint_indices=joint_indices,
        ee_link_index=ee_link_index,
        target_pos=pos_true,
        target_orn=orn_true,
        q_init=np.zeros_like(q_true),
        joint_lowers=joint_l,
        joint_uppers=joint_u,
    )

    pos_ik, _ = fk_pose(kuka_uid, joint_indices, ee_link_index, q_ik)
    print("True q:  ", q_true)
    print("IK q:    ", q_ik)
    print("pos_err: ", np.linalg.norm(pos_true - pos_ik))
    print("info:", info)

    p.disconnect(cid)
