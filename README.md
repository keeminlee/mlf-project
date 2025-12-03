# KUKA iiwa Inverse Kinematics Learning Project  
**ML Frontiers — Final Project**

This repository provides a complete pipeline for learning **inverse kinematics (IK)** for the 7-DoF **KUKA iiwa** robotic arm.

We include:

- Forward-kinematics–generated datasets (single-shot & trajectory)
- A baseline **MLP IK model** (single-shot & Δq trajectory mode)
- A **GNN IK model** that exploits the kinematic chain
- Tools for evaluating joint-space & end-effector errors
- Sequential **trajectory rollout** for long-horizon stability
- Classical IK solvers (Jacobian, Damped Least Squares)

---

## 🔧 Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📁 Project Structure

```text
mlf-project/
├── data/                          # CSV datasets generated via PyBullet
│   ├── kuka_fk_dataset.csv       # Single-shot FK dataset
│   └── kuka_traj_dataset_traj.csv # Trajectory dataset (Δq training)
│
├── src/                           # Source code
│   ├── kuka_fk_dataset.py        # FK data generator (single-shot & trajectory)
│   ├── classical_ik.py           # Classical IK: FK, Jacobian, DLS, PyBullet IK
│   ├── data_utils.py              # Shared data loading utilities
│   ├── checkpoint_utils.py        # Automatic checkpoint finding utilities
│   ├── mlp_ik.py                  # MLP IK model (single-shot & trajectory Δq)
│   ├── gnn_ik.py                  # GNN IK model (trajectory Δq only)
│   ├── eval_ik_models.py          # Model evaluation (joint + EE errors)
│   ├── trajectory_rollout.py      # Sequential trajectory rollout experiments
│   ├── grid_search.py             # Hyperparameter grid search utility
│   └── generate_report_results.py # Report generation (plots + metrics)
│
├── results/                        # Generated results and plots
│   ├── plots/                      # All plots (PNG, 300 DPI)
│   └── data/                       # Metrics and data files (JSON, NPZ)
│
├── mlp_ik_checkpoints/            # MLP single-shot checkpoints (created during training, ignored by git)
├── mlp_ik_traj_checkpoints/       # MLP trajectory Δq checkpoints (created during training, ignored by git)
├── gnn_ik_checkpoints/            # GNN trajectory Δq checkpoints (created during training, ignored by git)
│
├── run_report_generation.sh       # Script to generate all report results
├── train_lambda_sweep.sh          # Script to train models with different lambda values
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore patterns (checkpoints, logs, data files)
└── README.md                      # This file
```

**Note:** The following are ignored by git (via `.gitignore`) since they can be regenerated:
- `*_checkpoints/` directories (model checkpoints)
- `*_logs/` and `lightning_logs/` directories (training logs)
- `data/*.csv`, `data/*.npz`, `data/*.pkl` (generated datasets)

---

## 🏗️ 1. Generate Data

### **Single-shot FK dataset**

```bash
python src/kuka_fk_dataset.py \
  --num-samples 5000 \
  --include-orientation \
  --out-prefix data/kuka_fk_dataset
```

Produces:

- `data/kuka_fk_dataset.csv`

---

### **Trajectory dataset (Δq training)**

```bash
python src/kuka_fk_dataset.py \
  --data-type traj \
  --num-trajectories 200 \
  --steps-per-trajectory 50 \
  --include-orientation \
  --out-prefix data/kuka_traj_dataset
```

Produces:

- `data/kuka_traj_dataset_traj.csv`

Contains:
- end-effector pose  
- previous joint vector (`q_prev`)  
- next joint vector (`q_curr`)

---

## 🤖 2. Train MLP IK Model

### **A. Single-shot MLP**  
Input: pose → Output: absolute joint configuration

```bash
python src/mlp_ik.py \
  --csv-path data/kuka_fk_dataset.csv \
  --use-orientation \
  --batch-size 256 \
  --max-epochs 100 \
  --hidden-dims 256 256 128 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --dropout 0.1 \
  --accelerator auto
```

Checkpoints saved to:

- `mlp_ik_checkpoints/`

---

### **B. Trajectory Δq MLP**  
Input: [pose, q_prev] → Output: Δq

```bash
python src/mlp_ik.py \
  --csv-path data/kuka_traj_dataset_traj.csv \
  --use-orientation \
  --traj-mode \
  --batch-size 256 \
  --max-epochs 100 \
  --hidden-dims 256 256 128 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --dropout 0.1 \
  --accelerator auto \
  --lambda-movement 0.1
```

Checkpoints saved to:

- `mlp_ik_traj_checkpoints/`

---

## 🔗 3. Train GNN IK Model (Δq Only)

**Note:** The GNN model is designed for trajectory-based IK only (not single-shot). This is an architectural choice:
- The graph structure encodes the kinematic chain with joint nodes that include `q_prev` (previous joint state) as features
- The model predicts **Δq** (incremental joint changes) rather than absolute joint angles
- This enables smooth trajectory following and exploits the serial chain structure of the robot
- The movement penalty in the loss function encourages small, smooth joint movements

The MLP model is more flexible and supports both single-shot (absolute q) and trajectory (Δq) modes.

```bash
python src/gnn_ik.py \
  --csv-path data/kuka_traj_dataset_traj.csv \
  --use-orientation \
  --hidden-dim 64 \
  --num-layers 3 \
  --lambda-movement 0.1 \
  --batch-size 128 \
  --max-epochs 100 \
  --accelerator auto
```

Checkpoints saved to:

- `gnn_ik_checkpoints/`

---

## 📊 4. Evaluate MLP vs GNN  
(Joint error + End-Effector error)

```bash
# Simplest usage - auto-finds best checkpoints:
python src/eval_ik_models.py \
  --csv-path data/kuka_traj_dataset_traj.csv \
  --use-orientation \
  --num-samples 0

# Or manually specify checkpoints (optional):
python src/eval_ik_models.py \
  --csv-path data/kuka_traj_dataset_traj.csv \
  --use-orientation \
  --mlp-ckpt mlp_ik_traj_checkpoints/ikmlp-epoch=AAA-val_loss=BBB.ckpt \
  --gnn-ckpt gnn_ik_checkpoints/gnnik-epoch=XXX-val_loss=YYY.ckpt \
  --num-samples 0
```

**Note:** Evaluation uses the **TEST set** (same train/val/test split as training: seed=42, 15%/15% splits). The `--num-samples` parameter randomly samples from the test set. Set to `0` to evaluate on the entire test set.

Computes:

- Joint MSE / MAE  
- EE MSE / MAE (via FK)  
- Δq norms  
- Side-by-side comparison  

---

## 🌀 5. Sequential Trajectory Rollout

```bash
# Simplest usage - auto-finds best checkpoints:
python src/trajectory_rollout.py \
  --csv-path data/kuka_traj_dataset_traj.csv \
  --use-orientation \
  --num-trajectories 10 \
  --traj-length 30 \
  --device auto

# Or manually specify checkpoints (optional):
python src/trajectory_rollout.py \
  --csv-path data/kuka_traj_dataset_traj.csv \
  --use-orientation \
  --gnn-ckpt gnn_ik_checkpoints/gnnik-epoch=AAA-val_loss=BBB.ckpt \
  --mlp-ckpt mlp_ik_traj_checkpoints/ikmlp-epoch=XXX-val_loss=YYY.ckpt \
  --num-trajectories 10 \
  --traj-length 30 \
  --device auto
```

Outputs:

- mean & std EE error  
- Δq smoothness (L1/L2 norm)  
- long-horizon drift  

---

## 🧩 6. Classical IK (Baselines)

`src/classical_ik.py` provides:

- Forward kinematics  
- Jacobian computation  
- Damped Least Squares IK  
- PyBullet's built-in IK solver  

**Note:** Classical IK baselines are automatically evaluated and included in the report results. They are compared against MLP and GNN models in all plots and metrics.

---

## 📊 7. Generate Report Results and Plots

Generate all results and plots for the report. **Checkpoints are automatically found** - no need to manually specify paths!

```bash
# Simplest usage - auto-finds best checkpoints:
python src/generate_report_results.py \
  --csv-path data/kuka_traj_dataset_traj.csv \
  --results-dir results

# Or specify checkpoint directories:
python src/generate_report_results.py \
  --csv-path data/kuka_traj_dataset_traj.csv \
  --mlp-checkpoint-dir mlp_ik_traj_checkpoints \
  --gnn-checkpoint-dir gnn_ik_checkpoints \
  --results-dir results

# Or manually specify checkpoints (optional):
python src/generate_report_results.py \
  --csv-path data/kuka_traj_dataset_traj.csv \
  --mlp-ckpt mlp_ik_traj_checkpoints/ikmlp-epoch=AAA-val_loss=BBB.ckpt \
  --gnn-ckpt gnn_ik_checkpoints/gnnik-epoch=XXX-val_loss=YYY.ckpt \
  --results-dir results
```

**For full lambda sweep:**

1. **Train models with different lambda values:**
```bash
# Train all models with λ = 0.001, 0.01, 0.1, 1.0 (logarithmic scale: 1e-3, 1e-2, 1e-1, 1e0)
./train_lambda_sweep.sh
```

2. **Run lambda sweep analysis (auto-finds checkpoints):**
```bash
# The script automatically finds the best checkpoint for each lambda value!
python src/generate_report_results.py \
  --csv-path data/kuka_traj_dataset_traj.csv \
  --results-dir results
```

The script will:
- Automatically find the best checkpoint (lowest validation loss) for each lambda value
- Generate lambda sweep plots showing accuracy vs smoothness tradeoff
- No manual checkpoint path specification needed!

This generates:

1. **Trajectory rollout evaluation**: Sequential test where models predict full q₀ → q₁ → ... → qₜ trajectories
   - Shows stability and cumulative EE drift over long trajectories
   - Plot: EE error over trajectory steps (mean ± std across trajectories)
   - Plot: Cumulative EE drift over trajectories
   - Data: Statistics (mean/std/max EE error, cumulative drift, mean Δq norm)
   - **Great figure for the report!**

2. **Lambda sweep results**: Accuracy vs smoothness tradeoff
   - **For full sweep**: Train models with λ = 0.001, 0.01, 0.1, 1.0 (logarithmic scale: 1e-3, 1e-2, 1e-1, 1e0)
   - Use `./train_lambda_sweep.sh` to train all models automatically
   - Plot: Joint MSE vs Lambda (curve if multiple λ values, bar if single)
   - Plot: EE MSE vs Lambda
   - Plot: Mean Δq (smoothness) vs Lambda
   - Plot: Accuracy vs Smoothness tradeoff
   - Data: Metrics for each lambda value
   - **Should be super quick to run!**

3. **All plots for report** (includes classical IK baselines):
   - Joint MSE/MAE comparison (MLP, GNN, DLS, PyBullet)
   - EE MSE/MAE comparison (MLP, GNN, DLS, PyBullet)
   - Mean Δq per model (MLP, GNN, DLS, PyBullet)
   - Boxplot comparing movement magnitudes (MLP, GNN)

Results are saved to `results/` directory:
- `results/plots/` - All PNG plots (300 DPI, ready for report)
- `results/data/` - JSON and NPZ data files

---

## 📝 Notes

- PyBullet runs in DIRECT (headless) mode  
- All models implemented in PyTorch Lightning  
- GNN uses PyTorch Geometric  
- Supports CPU / CUDA / MPS (Apple Silicon)
- Train/val/test splits: 70%/15%/15% with seed=42 (consistent across training and evaluation)

