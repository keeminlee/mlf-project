#!/usr/bin/env python3
"""
generate_report_results.py

Generate all results and plots for the report:
1. Trajectory rollout evaluation (stability & cumulative EE drift)
2. Lambda sweep (accuracy vs smoothness tradeoff)
3. Comprehensive plots (Joint MSE/MAE, EE MSE/MAE, mean Δq, boxplots)

Results are saved to results/ directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src directory to path
_script_dir = Path(__file__).parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
import pandas as pd

from eval_ik_models import (
    eval_mlp_traj,
    eval_gnn_traj,
    setup_pybullet_fk,
    get_device,
)
from trajectory_rollout import rollout_model, build_gnn_input_graph
from data_utils import load_traj_csv
from mlp_ik import IKMLP
from gnn_ik import IK_GNN, KukaTrajGraphDataset
from torch_geometric.loader import DataLoader as GeoDataLoader
from classical_ik import (
    fk_position,
    ik_damped_least_squares,
    ik_pybullet_builtin,
    get_joint_limits,
)
import torch

# Set style for better plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11


def run_trajectory_rollout_eval(
    csv_path: str,
    mlp_ckpt: str,
    gnn_ckpt: str,
    num_trajectories: int = 20,
    traj_length: int = 50,
    device: str = "auto",
    results_dir: Path = Path("results"),
) -> Dict[str, Any]:
    """
    Run trajectory rollout evaluation and save results.
    Measures cumulative EE drift and stability over long trajectories.
    """
    print("\n" + "="*60)
    print("TRAJECTORY ROLLOUT EVALUATION")
    print("="*60)
    
    results_dir.mkdir(parents=True, exist_ok=True)
    data_dir = results_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    device_obj = get_device(device)
    
    # Load data
    poses, q_prev, q_curr, pose_dim = load_traj_csv(csv_path, use_orientation=True)
    N = poses.shape[0]
    
    # Setup PyBullet
    client_id, kuka_uid, joint_indices, ee_link_index = setup_pybullet_fk()
    
    # Load models
    mlp = IKMLP.load_from_checkpoint(str(mlp_ckpt), map_location="cpu")
    mlp.to(device_obj)
    
    gnn = IK_GNN.load_from_checkpoint(
        str(gnn_ckpt),
        map_location="cpu"
    )
    gnn.to(device_obj)
    
    # Sample trajectories
    rng = np.random.default_rng(42)
    all_results = {
        "mlp": {"ee_errors": [], "dq_norms": [], "dq_norms_L1": []},
        "gnn": {"ee_errors": [], "dq_norms": [], "dq_norms_L1": []},
    }
    
    trajectory_data = []
    
    for traj_idx in range(num_trajectories):
        # Sample a trajectory from the dataset
        start_idx = rng.integers(0, N - traj_length)
        poses_traj = poses[start_idx:start_idx + traj_length]
        q_init = q_prev[start_idx]
        
        # MLP rollout
        mlp_res = rollout_model(
            model_type="mlp",
            model=mlp,
            poses=poses_traj,
            q_init=q_init,
            pose_dim=pose_dim,
            device=device_obj,
            kuka_uid=kuka_uid,
            joint_indices=joint_indices,
            ee_link_index=ee_link_index,
        )
        
        # GNN rollout
        gnn_res = rollout_model(
            model_type="gnn",
            model=gnn,
            poses=poses_traj,
            q_init=q_init,
            pose_dim=pose_dim,
            device=device_obj,
            kuka_uid=kuka_uid,
            joint_indices=joint_indices,
            ee_link_index=ee_link_index,
        )
        
        # Store per-trajectory data
        trajectory_data.append({
            "traj_idx": traj_idx,
            "mlp_ee_errors": mlp_res["ee_errors"],
            "mlp_dq_norms": mlp_res["dq_norms"],
            "gnn_ee_errors": gnn_res["ee_errors"],
            "gnn_dq_norms": gnn_res["dq_norms"],
        })
        
        # Aggregate
        all_results["mlp"]["ee_errors"].extend(mlp_res["ee_errors"])
        all_results["mlp"]["dq_norms"].extend(mlp_res["dq_norms"])
        all_results["mlp"]["dq_norms_L1"].extend(mlp_res["dq_norms_L1"])
        all_results["gnn"]["ee_errors"].extend(gnn_res["ee_errors"])
        all_results["gnn"]["dq_norms"].extend(gnn_res["dq_norms"])
        all_results["gnn"]["dq_norms_L1"].extend(gnn_res["dq_norms_L1"])
    
    # Compute statistics
    stats = {}
    for model_name in ["mlp", "gnn"]:
        ee_errs = np.array(all_results[model_name]["ee_errors"])
        dq_norms = np.array(all_results[model_name]["dq_norms"])
        
        stats[model_name] = {
            "mean_ee_error": float(np.mean(ee_errs)),
            "std_ee_error": float(np.std(ee_errs)),
            "max_ee_error": float(np.max(ee_errs)),
            "mean_dq_norm": float(np.mean(dq_norms)),
            "std_dq_norm": float(np.std(dq_norms)),
            "cumulative_ee_drift": float(np.mean([np.sum(traj[f"{model_name}_ee_errors"]) 
                                                   for traj in trajectory_data])),
        }
    
    # Save results
    with open(data_dir / "trajectory_rollout_results.json", "w") as f:
        json.dump({
            "stats": stats,
            "num_trajectories": num_trajectories,
            "traj_length": traj_length,
        }, f, indent=2)
    
    # Save trajectory data for plotting
    np.savez_compressed(
        data_dir / "trajectory_rollout_data.npz",
        trajectory_data=trajectory_data,
    )
    
    # Generate trajectory rollout plots
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: EE error over trajectory steps (mean ± std across trajectories)
    fig, ax = plt.subplots(figsize=(10, 6))
    steps = np.arange(traj_length)
    
    mlp_ee_errors_all = np.array([traj["mlp_ee_errors"] for traj in trajectory_data])
    gnn_ee_errors_all = np.array([traj["gnn_ee_errors"] for traj in trajectory_data])
    
    mlp_mean = np.mean(mlp_ee_errors_all, axis=0)
    mlp_std = np.std(mlp_ee_errors_all, axis=0)
    gnn_mean = np.mean(gnn_ee_errors_all, axis=0)
    gnn_std = np.std(gnn_ee_errors_all, axis=0)
    
    ax.plot(steps, mlp_mean, label="MLP", color='steelblue', linewidth=2)
    ax.fill_between(steps, mlp_mean - mlp_std, mlp_mean + mlp_std, alpha=0.3, color='steelblue')
    
    ax.plot(steps, gnn_mean, label="GNN", color='coral', linewidth=2)
    ax.fill_between(steps, gnn_mean - gnn_std, gnn_mean + gnn_std, alpha=0.3, color='coral')
    
    ax.set_xlabel("Trajectory Step")
    ax.set_ylabel("EE Position Error (m)")
    ax.set_title("End-Effector Error Over Trajectory (Mean ± Std)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "trajectory_rollout_ee_error.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {plots_dir / 'trajectory_rollout_ee_error.png'}")
    
    # Plot 2: Cumulative EE drift over trajectories
    fig, ax = plt.subplots(figsize=(10, 6))
    
    mlp_cumulative = [np.cumsum(traj["mlp_ee_errors"]) for traj in trajectory_data]
    gnn_cumulative = [np.cumsum(traj["gnn_ee_errors"]) for traj in trajectory_data]
    
    mlp_cum_mean = np.mean(mlp_cumulative, axis=0)
    mlp_cum_std = np.std(mlp_cumulative, axis=0)
    gnn_cum_mean = np.mean(gnn_cumulative, axis=0)
    gnn_cum_std = np.std(gnn_cumulative, axis=0)
    
    ax.plot(steps, mlp_cum_mean, label="MLP", color='steelblue', linewidth=2)
    ax.fill_between(steps, mlp_cum_mean - mlp_cum_std, mlp_cum_mean + mlp_cum_std, alpha=0.3, color='steelblue')
    
    ax.plot(steps, gnn_cum_mean, label="GNN", color='coral', linewidth=2)
    ax.fill_between(steps, gnn_cum_mean - gnn_cum_std, gnn_cum_mean + gnn_cum_std, alpha=0.3, color='coral')
    
    ax.set_xlabel("Trajectory Step")
    ax.set_ylabel("Cumulative EE Drift (m)")
    ax.set_title("Cumulative End-Effector Drift Over Trajectory")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "trajectory_rollout_cumulative_drift.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {plots_dir / 'trajectory_rollout_cumulative_drift.png'}")
    
    print(f"\nResults saved to {data_dir / 'trajectory_rollout_results.json'}")
    print("\nSummary:")
    for model_name, model_stats in stats.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Mean EE Error: {model_stats['mean_ee_error']:.6f} ± {model_stats['std_ee_error']:.6f}")
        print(f"  Max EE Error: {model_stats['max_ee_error']:.6f}")
        print(f"  Cumulative EE Drift: {model_stats['cumulative_ee_drift']:.6f}")
        print(f"  Mean Δq Norm: {model_stats['mean_dq_norm']:.6f}")
    
    import pybullet as p
    p.disconnect(client_id)
    
    return all_results, trajectory_data


def run_lambda_sweep(
    csv_path: str,
    base_mlp_ckpt: str,
    base_gnn_ckpt: str,
    lambda_values: List[float] = [0.01, 0.1, 0.2],
    num_samples: int = 200,
    device: str = "auto",
    results_dir: Path = Path("results"),
) -> Dict[str, Any]:
    """
    Run lambda sweep to show accuracy vs smoothness tradeoff.
    Note: This requires retraining models with different lambda values.
    For now, we'll evaluate existing models and note lambda values.
    """
    print("\n" + "="*60)
    print("LAMBDA SWEEP EVALUATION")
    print("="*60)
    print("Note: This requires models trained with different lambda values.")
    print("Evaluating available checkpoints...")
    
    results_dir.mkdir(parents=True, exist_ok=True)
    data_dir = results_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    device_obj = get_device(device)
    client_id, kuka_uid, joint_indices, ee_link_index = setup_pybullet_fk()
    
    # Load base models to get their lambda values
    mlp = IKMLP.load_from_checkpoint(str(base_mlp_ckpt), map_location="cpu")
    gnn = IK_GNN.load_from_checkpoint(
        str(base_gnn_ckpt),
        map_location="cpu"
    )
    
    mlp_lambda = getattr(mlp.hparams, "lambda_movement", 0.1)
    gnn_lambda = getattr(gnn.hparams, "lambda_movement", 0.1)
    
    print(f"MLP lambda: {mlp_lambda}")
    print(f"GNN lambda: {gnn_lambda}")
    
    # Evaluate current models
    results = {}
    
    mlp_metrics = eval_mlp_traj(
        csv_path=csv_path,
        use_orientation=True,
        ckpt_path=base_mlp_ckpt,
        num_samples=num_samples,
        device=device_obj,
        kuka_uid=kuka_uid,
        joint_indices=joint_indices,
        ee_link_index=ee_link_index,
    )
    
    gnn_metrics = eval_gnn_traj(
        csv_path=csv_path,
        use_orientation=True,
        ckpt_path=base_gnn_ckpt,
        num_samples=num_samples,
        batch_size=64,
        device=device_obj,
        kuka_uid=kuka_uid,
        joint_indices=joint_indices,
        ee_link_index=ee_link_index,
    )
    
    results["mlp"] = {**mlp_metrics, "lambda": mlp_lambda}
    results["gnn"] = {**gnn_metrics, "lambda": gnn_lambda}
    
    # Save results
    with open(data_dir / "lambda_sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Generate lambda sweep plot (if we have multiple lambda values, otherwise just show current)
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract metrics for plotting
    mlp_lambda = results["mlp"]["lambda"]
    gnn_lambda = results["gnn"]["lambda"]
    
    # Create plot showing accuracy vs smoothness tradeoff
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Joint MSE vs Lambda
    ax1.scatter([mlp_lambda], [results["mlp"]["joint_mse"]], s=200, label="MLP", color='steelblue', marker='o', alpha=0.7)
    ax1.scatter([gnn_lambda], [results["gnn"]["joint_mse"]], s=200, label="GNN", color='coral', marker='s', alpha=0.7)
    ax1.set_xlabel("λ (Movement Penalty)")
    ax1.set_ylabel("Joint MSE")
    ax1.set_title("Joint MSE vs Lambda")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    
    # Plot 2: EE MSE vs Lambda
    ax2.scatter([mlp_lambda], [results["mlp"]["ee_mse"]], s=200, label="MLP", color='steelblue', marker='o', alpha=0.7)
    ax2.scatter([gnn_lambda], [results["gnn"]["ee_mse"]], s=200, label="GNN", color='coral', marker='s', alpha=0.7)
    ax2.set_xlabel("λ (Movement Penalty)")
    ax2.set_ylabel("EE MSE (m²)")
    ax2.set_title("End-Effector MSE vs Lambda")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    
    # Plot 3: Mean Δq L2 vs Lambda (smoothness)
    ax3.scatter([mlp_lambda], [results["mlp"]["mean_dq_L2"]], s=200, label="MLP", color='steelblue', marker='o', alpha=0.7)
    ax3.scatter([gnn_lambda], [results["gnn"]["mean_dq_L2"]], s=200, label="GNN", color='coral', marker='s', alpha=0.7)
    ax3.set_xlabel("λ (Movement Penalty)")
    ax3.set_ylabel("Mean Δq L2 Norm")
    ax3.set_title("Joint Movement (Smoothness) vs Lambda")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log')
    
    # Plot 4: Accuracy vs Smoothness tradeoff
    ax4.scatter([results["mlp"]["mean_dq_L2"]], [results["mlp"]["ee_mse"]], s=200, label="MLP", color='steelblue', marker='o', alpha=0.7)
    ax4.scatter([results["gnn"]["mean_dq_L2"]], [results["gnn"]["ee_mse"]], s=200, label="GNN", color='coral', marker='s', alpha=0.7)
    ax4.set_xlabel("Mean Δq L2 Norm (Smoothness)")
    ax4.set_ylabel("EE MSE (m²) (Accuracy)")
    ax4.set_title("Accuracy vs Smoothness Tradeoff")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    # Add annotations
    ax4.annotate(f'λ={mlp_lambda}', 
                (results["mlp"]["mean_dq_L2"], results["mlp"]["ee_mse"]),
                xytext=(5, 5), textcoords='offset points', fontsize=9)
    ax4.annotate(f'λ={gnn_lambda}', 
                (results["gnn"]["mean_dq_L2"], results["gnn"]["ee_mse"]),
                xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(plots_dir / "lambda_sweep_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {plots_dir / 'lambda_sweep_analysis.png'}")
    
    print(f"\nResults saved to {data_dir / 'lambda_sweep_results.json'}")
    print("\nNote: For full lambda sweep, train models with --lambda-movement 0.01, 0.1, 0.2")
    
    import pybullet as p
    p.disconnect(client_id)
    
    return results


def eval_classical_ik(
    csv_path: str,
    use_orientation: bool,
    num_samples: int,
    kuka_uid: int,
    joint_indices: List[int],
    ee_link_index: int,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Evaluate classical IK methods (DLS and PyBullet built-in) on test set.
    """
    poses, q_prev, q_curr, pose_dim = load_traj_csv(csv_path, use_orientation=use_orientation)
    N = poses.shape[0]
    
    # Use the same split as training
    rng = np.random.default_rng(seed)
    indices = np.arange(N)
    rng.shuffle(indices)
    
    n_val = int(val_frac * N)
    n_test = int(test_frac * N)
    n_train = N - n_val - n_test
    
    test_idx = indices[n_train + n_val:]
    
    poses_test = poses[test_idx]
    q_prev_test = q_prev[test_idx]
    q_curr_test = q_curr[test_idx]
    
    # Sample from test set if num_samples is specified
    if num_samples > 0 and num_samples < len(poses_test):
        rng_sample = np.random.default_rng(0)
        sample_idx = rng_sample.choice(len(poses_test), size=num_samples, replace=False)
        poses_sub = poses_test[sample_idx]
        q_prev_sub = q_prev_test[sample_idx]
        q_curr_sub = q_curr_test[sample_idx]
    else:
        poses_sub = poses_test
        q_prev_sub = q_prev_test
        q_curr_sub = q_curr_test
    
    # Get joint limits
    joint_lowers, joint_uppers = get_joint_limits(kuka_uid, joint_indices)
    
    # Evaluate DLS IK
    dls_joint_errors = []
    dls_ee_errors = []
    dls_dq_norms_L2 = []
    dls_dq_norms_L1 = []
    
    # Evaluate PyBullet built-in IK
    pybullet_joint_errors = []
    pybullet_ee_errors = []
    pybullet_dq_norms_L2 = []
    pybullet_dq_norms_L1 = []
    
    print(f"Evaluating classical IK on {len(poses_sub)} samples...")
    
    for i in range(len(poses_sub)):
        pose = poses_sub[i]
        q_prev_i = q_prev_sub[i]
        q_curr_i = q_curr_sub[i]
        
        target_pos = pose[:3]
        target_orn = pose[3:7] if use_orientation and pose_dim == 7 else None
        
        # DLS IK
        q_dls, info_dls = ik_damped_least_squares(
            body_uid=kuka_uid,
            joint_indices=joint_indices,
            ee_link_index=ee_link_index,
            target_pos=target_pos,
            target_orn=target_orn,
            q_init=q_prev_i,
            joint_lowers=joint_lowers,
            joint_uppers=joint_uppers,
            max_iters=50,
            pos_tol=1e-3,
            damping=1e-2,
        )
        
        # PyBullet built-in IK
        q_pb = ik_pybullet_builtin(
            body_uid=kuka_uid,
            joint_indices=joint_indices,
            ee_link_index=ee_link_index,
            target_pos=target_pos,
            target_orn=target_orn,
            joint_lowers=joint_lowers,
            joint_uppers=joint_uppers,
        )
        
        # Compute errors for DLS
        dls_joint_err = np.linalg.norm(q_dls - q_curr_i)
        dls_joint_errors.append(dls_joint_err)
        
        ee_dls = fk_position(kuka_uid, joint_indices, ee_link_index, q_dls)
        dls_ee_err = np.linalg.norm(ee_dls - target_pos)
        dls_ee_errors.append(dls_ee_err)
        
        dq_dls = q_dls - q_prev_i
        dls_dq_norms_L2.append(np.linalg.norm(dq_dls))
        dls_dq_norms_L1.append(np.sum(np.abs(dq_dls)))
        
        # Compute errors for PyBullet
        pb_joint_err = np.linalg.norm(q_pb - q_curr_i)
        pybullet_joint_errors.append(pb_joint_err)
        
        ee_pb = fk_position(kuka_uid, joint_indices, ee_link_index, q_pb)
        pb_ee_err = np.linalg.norm(ee_pb - target_pos)
        pybullet_ee_errors.append(pb_ee_err)
        
        dq_pb = q_pb - q_prev_i
        pybullet_dq_norms_L2.append(np.linalg.norm(dq_pb))
        pybullet_dq_norms_L1.append(np.sum(np.abs(dq_pb)))
    
    # Compute metrics
    dls_metrics = {
        "joint_mse": float(np.mean(np.array(dls_joint_errors) ** 2)),
        "joint_mae": float(np.mean(dls_joint_errors)),
        "ee_mse": float(np.mean(np.array(dls_ee_errors) ** 2)),
        "ee_mae": float(np.mean(dls_ee_errors)),
        "mean_dq_L2": float(np.mean(dls_dq_norms_L2)),
        "mean_dq_L1": float(np.mean(dls_dq_norms_L1)),
    }
    
    pybullet_metrics = {
        "joint_mse": float(np.mean(np.array(pybullet_joint_errors) ** 2)),
        "joint_mae": float(np.mean(pybullet_joint_errors)),
        "ee_mse": float(np.mean(np.array(pybullet_ee_errors) ** 2)),
        "ee_mae": float(np.mean(pybullet_ee_errors)),
        "mean_dq_L2": float(np.mean(pybullet_dq_norms_L2)),
        "mean_dq_L1": float(np.mean(pybullet_dq_norms_L1)),
    }
    
    return {
        "dls": dls_metrics,
        "pybullet": pybullet_metrics,
    }


def generate_all_plots(
    csv_path: str,
    mlp_ckpt: str,
    gnn_ckpt: str,
    num_samples: int = 500,
    device: str = "auto",
    results_dir: Path = Path("results"),
    include_classical: bool = True,
) -> None:
    """
    Generate all plots for the report:
    - Joint MSE/MAE
    - EE MSE/MAE
    - mean Δq per model
    - Boxplot comparing movement magnitudes
    """
    print("\n" + "="*60)
    print("GENERATING PLOTS FOR REPORT")
    print("="*60)
    
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    data_dir = results_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    device_obj = get_device(device)
    client_id, kuka_uid, joint_indices, ee_link_index = setup_pybullet_fk()
    
    # Evaluate models
    print("\nEvaluating models...")
    mlp_metrics = eval_mlp_traj(
        csv_path=csv_path,
        use_orientation=True,
        ckpt_path=mlp_ckpt,
        num_samples=num_samples,
        device=device_obj,
        kuka_uid=kuka_uid,
        joint_indices=joint_indices,
        ee_link_index=ee_link_index,
    )
    
    gnn_metrics = eval_gnn_traj(
        csv_path=csv_path,
        use_orientation=True,
        ckpt_path=gnn_ckpt,
        num_samples=num_samples,
        batch_size=64,
        device=device_obj,
        kuka_uid=kuka_uid,
        joint_indices=joint_indices,
        ee_link_index=ee_link_index,
    )
    
    # Evaluate classical IK baselines
    classical_metrics = {}
    if include_classical:
        print("\nEvaluating classical IK baselines...")
        classical_metrics = eval_classical_ik(
            csv_path=csv_path,
            use_orientation=True,
            num_samples=num_samples,
            kuka_uid=kuka_uid,
            joint_indices=joint_indices,
            ee_link_index=ee_link_index,
        )
    
    # Prepare data for plotting
    models = ["MLP", "GNN"]
    joint_mse_vals = [mlp_metrics["joint_mse"], gnn_metrics["joint_mse"]]
    joint_mae_vals = [mlp_metrics["joint_mae"], gnn_metrics["joint_mae"]]
    ee_mse_vals = [mlp_metrics["ee_mse"], gnn_metrics["ee_mse"]]
    ee_mae_vals = [mlp_metrics["ee_mae"], gnn_metrics["ee_mae"]]
    dq_l2_vals = [mlp_metrics["mean_dq_L2"], gnn_metrics["mean_dq_L2"]]
    dq_l1_vals = [mlp_metrics["mean_dq_L1"], gnn_metrics["mean_dq_L1"]]
    
    if include_classical and classical_metrics:
        models.extend(["DLS", "PyBullet"])
        joint_mse_vals.extend([classical_metrics["dls"]["joint_mse"], classical_metrics["pybullet"]["joint_mse"]])
        joint_mae_vals.extend([classical_metrics["dls"]["joint_mae"], classical_metrics["pybullet"]["joint_mae"]])
        ee_mse_vals.extend([classical_metrics["dls"]["ee_mse"], classical_metrics["pybullet"]["ee_mse"]])
        ee_mae_vals.extend([classical_metrics["dls"]["ee_mae"], classical_metrics["pybullet"]["ee_mae"]])
        dq_l2_vals.extend([classical_metrics["dls"]["mean_dq_L2"], classical_metrics["pybullet"]["mean_dq_L2"]])
        dq_l1_vals.extend([classical_metrics["dls"]["mean_dq_L1"], classical_metrics["pybullet"]["mean_dq_L1"]])
    
    plot_data = {
        "Model": models,
        "Joint MSE": joint_mse_vals,
        "Joint MAE": joint_mae_vals,
        "EE MSE": ee_mse_vals,
        "EE MAE": ee_mae_vals,
        "Mean Δq L2": dq_l2_vals,
        "Mean Δq L1": dq_l1_vals,
    }
    
    df = pd.DataFrame(plot_data)
    
    # Save metrics data
    metrics_dict = {"mlp": mlp_metrics, "gnn": gnn_metrics}
    if include_classical and classical_metrics:
        metrics_dict.update(classical_metrics)
    with open(data_dir / "evaluation_metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=2)
    
    # 1. Joint MSE/MAE plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(df))
    width = 0.6 if len(df) <= 2 else 0.5
    
    # Color scheme
    colors = ['steelblue', 'coral', 'green', 'orange']
    if len(df) > 4:
        colors = colors + ['purple', 'brown', 'pink', 'gray']
    
    # Plot all models
    for i, model in enumerate(df["Model"]):
        color = colors[i % len(colors)]
        mse_val = df.loc[df["Model"] == model, "Joint MSE"].values[0]
        mae_val = df.loc[df["Model"] == model, "Joint MAE"].values[0]
        ax1.bar([i], [mse_val], width, label=model, alpha=0.8, color=color)
        ax2.bar([i], [mae_val], width, label=model, alpha=0.8, color=color)
    
    ax1.set_ylabel("Joint MSE")
    ax1.set_title("Joint Mean Squared Error")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["Model"])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")
    
    ax2.set_ylabel("Joint MAE")
    ax2.set_title("Joint Mean Absolute Error")
    ax2.set_xticks(x)
    ax2.set_xticklabels(df["Model"])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(plots_dir / "joint_mse_mae.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {plots_dir / 'joint_mse_mae.png'}")
    
    # 2. EE MSE/MAE plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot all models
    for i, model in enumerate(df["Model"]):
        color = colors[i % len(colors)]
        ee_mse_val = df.loc[df["Model"] == model, "EE MSE"].values[0]
        ee_mae_val = df.loc[df["Model"] == model, "EE MAE"].values[0]
        ax1.bar([i], [ee_mse_val], width, label=model, alpha=0.8, color=color)
        ax2.bar([i], [ee_mae_val], width, label=model, alpha=0.8, color=color)
    
    ax1.set_ylabel("EE MSE (m²)")
    ax1.set_title("End-Effector Mean Squared Error")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["Model"])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")
    
    ax2.set_ylabel("EE MAE (m)")
    ax2.set_title("End-Effector Mean Absolute Error")
    ax2.set_xticks(x)
    ax2.set_xticklabels(df["Model"])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(plots_dir / "ee_mse_mae.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {plots_dir / 'ee_mse_mae.png'}")
    
    # 3. Mean Δq per model
    fig, ax = plt.subplots(figsize=(12, 6))
    x_pos = np.arange(len(df))
    width = 0.35
    
    # Extract values for all models
    dq_l2_vals = [df.loc[df["Model"] == model, "Mean Δq L2"].values[0] for model in df["Model"]]
    dq_l1_vals = [df.loc[df["Model"] == model, "Mean Δq L1"].values[0] for model in df["Model"]]
    
    ax.bar(x_pos - width/2, dq_l2_vals, width, label="L2 Norm", alpha=0.8, color='steelblue')
    ax.bar(x_pos + width/2, dq_l1_vals, width, label="L1 Norm", alpha=0.8, color='coral')
    ax.set_ylabel("Mean Δq")
    ax.set_title("Mean Joint Movement (Δq) per Model")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df["Model"])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(plots_dir / "mean_dq_per_model.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {plots_dir / 'mean_dq_per_model.png'}")
    
    # 4. Boxplot comparing movement magnitudes
    # For this, we need per-sample data
    print("\nCollecting per-sample data for boxplot...")
    
    # Get per-sample dq_norms by running detailed evaluation
    # We'll use a helper function to collect per-sample metrics
    poses, q_prev, q_curr, pose_dim = load_traj_csv(csv_path, use_orientation=True)
    N = poses.shape[0]
    
    # Use test split
    rng = np.random.default_rng(42)
    indices = np.arange(N)
    rng.shuffle(indices)
    n_val = int(0.15 * N)
    n_test = int(0.15 * N)
    n_train = N - n_val - n_test
    test_idx = indices[n_train + n_val:]
    
    poses_test = poses[test_idx]
    q_prev_test = q_prev[test_idx]
    q_curr_test = q_curr[test_idx]
    
    # Sample for boxplot
    if num_samples > 0 and num_samples < len(poses_test):
        sample_idx = np.random.choice(len(poses_test), size=num_samples, replace=False)
        poses_sub = poses_test[sample_idx]
        q_prev_sub = q_prev_test[sample_idx]
        q_curr_sub = q_curr_test[sample_idx]
    else:
        poses_sub = poses_test
        q_prev_sub = q_prev_test
        q_curr_sub = q_curr_test
    
    # MLP predictions
    mlp_model = IKMLP.load_from_checkpoint(str(mlp_ckpt), map_location="cpu")
    mlp_model.to(device_obj)
    mlp_model.eval()
    
    predict_delta = bool(getattr(mlp_model.hparams, "predict_delta", False))
    input_dim = int(getattr(mlp_model.hparams, "input_dim", pose_dim))
    
    if input_dim == pose_dim + 7:
        X_mlp = np.concatenate([poses_sub, q_prev_sub], axis=1).astype(np.float32)
    else:
        X_mlp = poses_sub.astype(np.float32)
    
    with torch.no_grad():
        x_t = torch.from_numpy(X_mlp).to(device_obj)
        out = mlp_model(x_t).cpu().numpy().astype(np.float32)
    
    if predict_delta:
        dq_mlp = out
    else:
        dq_mlp = out - q_prev_sub
    
    mlp_dq_samples = np.linalg.norm(dq_mlp, axis=1)
    
    # GNN predictions
    gnn_model = IK_GNN.load_from_checkpoint(str(gnn_ckpt), map_location="cpu")
    gnn_model.to(device_obj)
    gnn_model.eval()
    
    dataset = KukaTrajGraphDataset(poses_sub, q_prev_sub, q_curr_sub, pose_dim)
    loader = GeoDataLoader(dataset, batch_size=64, shuffle=False)
    
    all_dq_gnn = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device_obj)
            dq_pred = gnn_model(batch)
            dq_norms = torch.norm(dq_pred, dim=1).cpu().numpy()
            all_dq_gnn.extend(dq_norms)
    
    gnn_dq_samples = np.array(all_dq_gnn)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    box_data = [mlp_dq_samples, gnn_dq_samples]
    bp = ax.boxplot(box_data, labels=["MLP", "GNN"], patch_artist=True)
    
    colors = ['lightblue', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel("Δq L2 Norm")
    ax.set_title("Distribution of Joint Movement Magnitudes")
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(plots_dir / "movement_magnitude_boxplot.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {plots_dir / 'movement_magnitude_boxplot.png'}")
    
    # Save plot data
    with open(data_dir / "plot_data.json", "w") as f:
        json.dump(plot_data, f, indent=2)
    
    import pybullet as p
    p.disconnect(client_id)
    
    print(f"\nAll plots saved to {plots_dir}/")
    print(f"All data saved to {data_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Generate all results and plots for the report"
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="data/kuka_traj_dataset_traj.csv",
        help="Path to trajectory CSV",
    )
    parser.add_argument(
        "--mlp-ckpt",
        type=str,
        required=True,
        help="Path to MLP checkpoint",
    )
    parser.add_argument(
        "--gnn-ckpt",
        type=str,
        required=True,
        help="Path to GNN checkpoint",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--skip-rollout",
        action="store_true",
        help="Skip trajectory rollout evaluation",
    )
    parser.add_argument(
        "--skip-lambda",
        action="store_true",
        help="Skip lambda sweep",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip plot generation",
    )
    parser.add_argument(
        "--no-classical",
        action="store_true",
        help="Skip classical IK baseline evaluation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: auto, cpu, cuda, mps",
    )
    
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    
    if not args.skip_rollout:
        run_trajectory_rollout_eval(
            csv_path=args.csv_path,
            mlp_ckpt=args.mlp_ckpt,
            gnn_ckpt=args.gnn_ckpt,
            num_trajectories=20,
            traj_length=50,
            device=args.device,
            results_dir=results_dir,
        )
    
    if not args.skip_lambda:
        run_lambda_sweep(
            csv_path=args.csv_path,
            base_mlp_ckpt=args.mlp_ckpt,
            base_gnn_ckpt=args.gnn_ckpt,
            device=args.device,
            results_dir=results_dir,
        )
    
    if not args.skip_plots:
        generate_all_plots(
            csv_path=args.csv_path,
            mlp_ckpt=args.mlp_ckpt,
            gnn_ckpt=args.gnn_ckpt,
            num_samples=500,
            device=args.device,
            results_dir=results_dir,
            include_classical=not args.no_classical,
        )
    
    print("\n" + "="*60)
    print("ALL RESULTS GENERATED SUCCESSFULLY!")
    print("="*60)
    print(f"Results directory: {results_dir.absolute()}")
    print(f"  - Plots: {results_dir / 'plots'}")
    print(f"  - Data: {results_dir / 'data'}")


if __name__ == "__main__":
    main()

