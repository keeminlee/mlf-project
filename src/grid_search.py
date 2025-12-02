from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, List

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


class IKGridSearch:
    """
    Generic grid search runner for IK regression models (MLP, GNN, etc.).

    model_builder:      (params, n_joints) -> pl.LightningModule
    datamodule_builder: (params, splits)   -> pl.LightningDataModule

    The runner:
    - generates all parameter combinations
    - trains each model using Lightning
    - logs best scores + paths
    - saves grid_search_results.json
    """

    def __init__(
        self,
        output_dir: str | Path,
        param_grid: Dict[str, List[Any]],
        model_builder: Callable[[Dict[str, Any], int], pl.LightningModule],
        datamodule_builder: Callable[[Dict[str, Any], Dict[str, Any]], pl.LightningDataModule],

        monitor_metric: str = "val_loss",
        mode: str = "min",
        patience: int = 15,
        max_epochs: int = 100,
        accelerator: str = "auto",

        n_joints: int = 7,   # ← IK-specific (better name than n_keypoints)
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.param_grid = param_grid
        self.model_builder = model_builder
        self.datamodule_builder = datamodule_builder

        self.monitor_metric = monitor_metric
        self.mode = mode
        self.patience = patience
        self.max_epochs = max_epochs
        self.accelerator = accelerator

        self.n_joints = int(n_joints)

    # ------------------------------------------------------------------
    # Generate all parameter combinations
    # ------------------------------------------------------------------

    def generate_param_combinations(self) -> List[Dict[str, Any]]:
        import itertools

        keys = list(self.param_grid.keys())
        values = [self.param_grid[k] for k in keys]

        combos: List[Dict[str, Any]] = []

        for vtuple in itertools.product(*values):
            d = dict(zip(keys, vtuple))
            # Create short hash slug for checkpoint naming
            slug = hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest()[:10]
            d["_slug"] = slug
            combos.append(d)

        return combos

    # ------------------------------------------------------------------
    # Run all combinations
    # ------------------------------------------------------------------

    def run(self, splits: Dict[str, Any]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []

        for params in self.generate_param_combinations():
            try:
                res = self._run_single(params, splits)
                results.append(res)
                self._save_results(results)
            except Exception as e:
                print(f"[IKGridSearch] Failed for params {params}: {e}")

        return results

    # ------------------------------------------------------------------
    # Train one model with one parameter combination
    # ------------------------------------------------------------------

    def _run_single(self, params: Dict[str, Any], splits: Dict[str, Any]) -> Dict[str, Any]:
        # Build model + datamodule
        model = self.model_builder(params, self.n_joints)
        datamodule = self.datamodule_builder(params, splits)

        # Checkpoint directory
        ckpt_dir = self.output_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_cb = ModelCheckpoint(
            monitor=self.monitor_metric,
            dirpath=ckpt_dir,
            filename=f"model-{params['_slug']}",
            save_top_k=1,
            mode=self.mode,
        )

        early_cb = EarlyStopping(
            monitor=self.monitor_metric,
            patience=self.patience,
            mode=self.mode,
        )

        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            callbacks=[checkpoint_cb, early_cb],
            enable_progress_bar=True,
            log_every_n_steps=10,
            accelerator=self.accelerator,
        )

        trainer.fit(model, datamodule)

        clean_params = {k: v for k, v in params.items() if k != "_slug"}

        return {
            "params": clean_params,
            "best_score": float(checkpoint_cb.best_model_score.item()),
            "best_path": str(checkpoint_cb.best_model_path),
        }

    # ------------------------------------------------------------------
    # Save results after each run for safety
    # ------------------------------------------------------------------

    def _save_results(self, results: List[Dict[str, Any]]) -> None:
        serializable = [
            {
                "params": r["params"],
                "best_score": float(r["best_score"]),
                "best_path": r["best_path"],
            }
            for r in results
        ]

        serializable.sort(key=lambda x: x["best_score"])

        with open(self.output_dir / "grid_search_results.json", "w") as f:
            json.dump(serializable, f, indent=2)
