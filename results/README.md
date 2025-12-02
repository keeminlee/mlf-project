# Report Results Directory

This directory contains all generated results and plots for the report.

## Directory Structure

```
results/
├── plots/          # All generated plots (PNG, 300 DPI, ready for report)
│   ├── joint_mse_mae.png
│   ├── ee_mse_mae.png
│   ├── mean_dq_per_model.png
│   └── movement_magnitude_boxplot.png
│
└── data/           # All data files (JSON, NPZ)
    ├── trajectory_rollout_results.json
    ├── trajectory_rollout_data.npz
    ├── lambda_sweep_results.json
    ├── evaluation_metrics.json
    └── plot_data.json
```

## Generated Content

### 1. Trajectory Rollout Evaluation
- **File**: `data/trajectory_rollout_results.json`
- **Description**: Sequential trajectory evaluation showing stability and cumulative EE drift
- **Metrics**: Mean/std/max EE error, cumulative drift, mean Δq norms
- **Configuration**: 20 trajectories, 50 steps each

### 2. Lambda Sweep Results
- **File**: `data/lambda_sweep_results.json`
- **Description**: Accuracy vs smoothness tradeoff for different lambda values
- **Note**: Requires models trained with different λ values (0.01, 0.1, 0.2)

### 3. Evaluation Metrics
- **File**: `data/evaluation_metrics.json`
- **Description**: Comprehensive metrics for MLP and GNN models
- **Includes**: Joint MSE/MAE, EE MSE/MAE, mean Δq (L1/L2)

### 4. Plots for Report

#### `plots/joint_mse_mae.png`
- Joint space error comparison (MSE and MAE)
- Side-by-side bar charts for MLP vs GNN

#### `plots/ee_mse_mae.png`
- End-effector error comparison (MSE and MAE)
- Side-by-side bar charts for MLP vs GNN

#### `plots/mean_dq_per_model.png`
- Mean joint movement (Δq) per model
- Shows L1 and L2 norms

#### `plots/movement_magnitude_boxplot.png`
- Distribution of joint movement magnitudes
- Boxplot comparing MLP vs GNN
- Shows spread and outliers

## Usage

To regenerate all results:

```bash
./run_report_generation.sh
```

Or manually:

```bash
python src/generate_report_results.py \
  --csv-path data/kuka_traj_dataset_traj.csv \
  --mlp-ckpt mlp_ik_traj_checkpoints/ikmlp-epoch=012-val_loss=0.0025.ckpt \
  --gnn-ckpt gnn_ik_checkpoints/gnnik-epoch=019-val_loss=0.0025.ckpt \
  --results-dir results
```

## Notes

- All plots are saved at 300 DPI for publication quality
- JSON files contain all numerical results for further analysis
- NPZ files contain raw trajectory data for custom plotting
- Update checkpoint paths in `run_report_generation.sh` before running

