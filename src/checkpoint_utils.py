"""
checkpoint_utils.py

Utilities for automatically finding and managing model checkpoints.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, List, Tuple
import re
import glob
import torch


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
        # Match digits and decimal point, but stop before .ckpt extension
        match = re.search(r'val_loss=(\d+\.?\d*)', ckpt.name)
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
    Find best checkpoints for each lambda value by extracting lambda from checkpoint hparams.
    
    Strategy:
    1. First try subdirectories: checkpoint_dir/lambda_0.001/, etc.
    2. If not found, scan all checkpoints in main directory, extract lambda from hparams,
       and find the best checkpoint for each lambda value.
    
    Args:
        checkpoint_dir: Base directory containing checkpoints
        model_type: "mlp" or "gnn"
        lambda_values: List of lambda values to find checkpoints for
    
    Returns:
        Dict mapping lambda values to checkpoint paths (or None if not found)
    """
    checkpoint_dir = Path(checkpoint_dir)
    results = {lam: None for lam in lambda_values}
    
    # First, try subdirectory approach
    all_found_in_subdirs = True
    for lam in lambda_values:
        lambda_subdir = checkpoint_dir / f"lambda_{lam}"
        if lambda_subdir.exists():
            ckpt = find_best_checkpoint(lambda_subdir, model_type)
            if ckpt:
                results[lam] = ckpt
                continue
        all_found_in_subdirs = False
    
    if all_found_in_subdirs and all(results.values()):
        return results
    
    # Fall back to scanning all checkpoints and extracting lambda from hparams
    if model_type.lower() == "mlp":
        pattern = "ikmlp-epoch=*-val_loss=*.ckpt"
    elif model_type.lower() == "gnn":
        pattern = "gnnik-epoch=*-val_loss=*.ckpt"
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    checkpoints = list(checkpoint_dir.glob(pattern))
    
    if not checkpoints:
        return results
    
    # Group checkpoints by lambda value extracted from hparams
    # Import here to avoid circular imports
    if model_type.lower() == "mlp":
        from mlp_ik import IKMLP as ModelClass
    else:
        from gnn_ik import IK_GNN as ModelClass
    
    checkpoints_by_lambda: Dict[float, List[Tuple[str, float]]] = {}
    
    for ckpt_path in checkpoints:
        try:
            # Load checkpoint to extract lambda
            ckpt = ModelClass.load_from_checkpoint(str(ckpt_path), map_location="cpu")
            ckpt_lambda = getattr(ckpt.hparams, "lambda_movement", None)
            
            if ckpt_lambda is None:
                continue
            
            # Round to nearest lambda value in our list (handles floating point issues)
            matched_lambda = None
            for target_lambda in lambda_values:
                if abs(ckpt_lambda - target_lambda) < 0.0001:
                    matched_lambda = target_lambda
                    break
            
            if matched_lambda is None:
                continue
            
            # Extract validation loss from filename
            match = re.search(r'val_loss=(\d+\.?\d*)', ckpt_path.name)
            if match:
                val_loss = float(match.group(1))
                
                if matched_lambda not in checkpoints_by_lambda:
                    checkpoints_by_lambda[matched_lambda] = []
                checkpoints_by_lambda[matched_lambda].append((str(ckpt_path), val_loss))
        except Exception:
            # Skip checkpoints that can't be loaded
            continue
    
    # Find best checkpoint (lowest val_loss) for each lambda
    for lam in lambda_values:
        if lam not in checkpoints_by_lambda:
            continue
        
        # Sort by validation loss (ascending) and take the best
        checkpoints_by_lambda[lam].sort(key=lambda x: x[1])
        results[lam] = checkpoints_by_lambda[lam][0][0]
    
    return results


def find_all_lambda_checkpoints(
    mlp_checkpoint_dir: str | Path = "mlp_ik_traj_checkpoints",
    gnn_checkpoint_dir: str | Path = "gnn_ik_checkpoints",
    lambda_values: List[float] = [0.001, 0.01, 0.1, 1.0],
) -> Tuple[Dict[float, Optional[str]], Dict[float, Optional[str]]]:
    """
    Find checkpoints for all lambda values for both MLP and GNN.
    
    Uses find_checkpoints_by_lambda which:
    1. First tries subdirectories: checkpoint_dir/lambda_0.001/, etc.
    2. If not found, scans all checkpoints and extracts lambda from hparams
       to find the best checkpoint for each lambda value
    
    Returns:
        Tuple of (mlp_checkpoints_dict, gnn_checkpoints_dict)
    """
    mlp_results = find_checkpoints_by_lambda(
        mlp_checkpoint_dir, "mlp", lambda_values
    )
    gnn_results = find_checkpoints_by_lambda(
        gnn_checkpoint_dir, "gnn", lambda_values
    )
    
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

