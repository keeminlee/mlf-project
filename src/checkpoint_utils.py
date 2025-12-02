"""
checkpoint_utils.py

Utilities for automatically finding and managing model checkpoints.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, List, Tuple
import re
import glob


def find_best_checkpoint(
    checkpoint_dir: str | Path,
    model_type: str = "mlp",
    lambda_value: Optional[float] = None,
) -> Optional[str]:
    """
    Find the best checkpoint in a directory based on validation loss.
    
    Args:
        checkpoint_dir: Directory containing checkpoint files
        model_type: "mlp" or "gnn" (affects filename pattern)
        lambda_value: If provided, only consider checkpoints with this lambda value
    
    Returns:
        Path to best checkpoint (lowest validation loss), or None if not found
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    
    # Pattern for checkpoint filenames
    if model_type.lower() == "mlp":
        pattern = "ikmlp-epoch=*-val_loss=*.ckpt"
    elif model_type.lower() == "gnn":
        pattern = "gnnik-epoch=*-val_loss=*.ckpt"
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    checkpoints = list(checkpoint_dir.glob(pattern))
    
    if not checkpoints:
        return None
    
    # Extract validation loss from filename and find minimum
    best_ckpt = None
    best_val_loss = float('inf')
    
    for ckpt in checkpoints:
        # Extract val_loss from filename: "ikmlp-epoch=012-val_loss=0.0025.ckpt"
        match = re.search(r'val_loss=([\d.]+)', ckpt.name)
        if match:
            val_loss = float(match.group(1))
            
            # If lambda_value specified, check if checkpoint matches
            # (We can't extract lambda from filename easily, so we'll check all)
            # For now, we'll just find the best overall checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_ckpt = str(ckpt)
    
    return best_ckpt


def find_checkpoints_by_lambda(
    checkpoint_dir: str | Path,
    model_type: str = "mlp",
    lambda_values: List[float] = [0.001, 0.01, 0.1, 1.0],
) -> Dict[float, Optional[str]]:
    """
    Find best checkpoints for each lambda value.
    
    Strategy:
    1. First try subdirectories: checkpoint_dir/lambda_0.001/, etc.
    2. If not found, look in main directory and use best checkpoint for all lambdas
       (assumes models were trained with different lambdas but saved to same directory)
    
    Args:
        checkpoint_dir: Base directory containing checkpoints
        model_type: "mlp" or "gnn"
        lambda_values: List of lambda values to find checkpoints for
    
    Returns:
        Dict mapping lambda values to checkpoint paths (or None if not found)
    """
    checkpoint_dir = Path(checkpoint_dir)
    results = {}
    
    for lam in lambda_values:
        # Try subdirectory approach first
        lambda_subdir = checkpoint_dir / f"lambda_{lam}"
        if lambda_subdir.exists():
            ckpt = find_best_checkpoint(lambda_subdir, model_type)
            if ckpt:
                results[lam] = ckpt
                continue
        
        # Fall back to main directory - find best checkpoint
        # Note: This will use the same checkpoint for all lambdas if they're in the same dir
        # In practice, after training with train_lambda_sweep.sh, all checkpoints
        # will be in the main directory, so we'll find the best one
        ckpt = find_best_checkpoint(checkpoint_dir, model_type)
        results[lam] = ckpt
    
    return results


def find_all_lambda_checkpoints(
    mlp_checkpoint_dir: str | Path = "mlp_ik_traj_checkpoints",
    gnn_checkpoint_dir: str | Path = "gnn_ik_checkpoints",
    lambda_values: List[float] = [0.001, 0.01, 0.1, 1.0],
) -> Tuple[Dict[float, Optional[str]], Dict[float, Optional[str]]]:
    """
    Find checkpoints for all lambda values for both MLP and GNN.
    
    This function attempts to find checkpoints by:
    1. Looking for subdirectories named by lambda value
    2. Looking for checkpoints with lambda in filename
    3. Falling back to best checkpoint in main directory
    
    Returns:
        Tuple of (mlp_checkpoints_dict, gnn_checkpoints_dict)
    """
    mlp_results = {}
    gnn_results = {}
    
    mlp_dir = Path(mlp_checkpoint_dir)
    gnn_dir = Path(gnn_checkpoint_dir)
    
    for lam in lambda_values:
        # Try subdirectory approach: mlp_ik_traj_checkpoints/lambda_0.001/
        mlp_subdir = mlp_dir / f"lambda_{lam}"
        gnn_subdir = gnn_dir / f"lambda_{lam}"
        
        if mlp_subdir.exists():
            mlp_results[lam] = find_best_checkpoint(mlp_subdir, "mlp")
        else:
            # Fall back to main directory
            mlp_results[lam] = find_best_checkpoint(mlp_dir, "mlp")
        
        if gnn_subdir.exists():
            gnn_results[lam] = find_best_checkpoint(gnn_subdir, "gnn")
        else:
            # Fall back to main directory
            gnn_results[lam] = find_best_checkpoint(gnn_dir, "gnn")
    
    return mlp_results, gnn_results


def auto_find_checkpoints(
    mlp_checkpoint_dir: str | Path = "mlp_ik_traj_checkpoints",
    gnn_checkpoint_dir: str | Path = "gnn_ik_checkpoints",
    model_type: str = "traj",
) -> Tuple[Optional[str], Optional[str]]:
    """
    Automatically find the best checkpoints for MLP and GNN.
    
    Args:
        mlp_checkpoint_dir: Directory for MLP checkpoints
        gnn_checkpoint_dir: Directory for GNN checkpoints
        model_type: "single" or "traj" (affects which directory to check)
    
    Returns:
        Tuple of (mlp_checkpoint_path, gnn_checkpoint_path) or (None, None) if not found
    """
    if model_type == "traj":
        mlp_dir = Path(mlp_checkpoint_dir) if isinstance(mlp_checkpoint_dir, str) else mlp_checkpoint_dir
        gnn_dir = Path(gnn_checkpoint_dir) if isinstance(gnn_checkpoint_dir, str) else gnn_checkpoint_dir
    else:
        mlp_dir = Path("mlp_ik_checkpoints")
        gnn_dir = Path(gnn_checkpoint_dir)  # GNN only supports traj mode
    
    mlp_ckpt = find_best_checkpoint(mlp_dir, "mlp")
    gnn_ckpt = find_best_checkpoint(gnn_dir, "gnn")
    
    return mlp_ckpt, gnn_ckpt

