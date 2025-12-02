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
data/                     # CSV datasets generated via PyBullet
src/
  kuka_fk_dataset.py      # FK data generator
  classical_ik.py         # FK, Jacobian, DLS IK, PyBullet IK
  data_utils.py           # Shared data loading utilities
  mlp_ik.py               # MLP IK model
  gnn_ik.py               # GNN Δq model
  eval_ik_models.py       # Evaluation (joint + EE)
  trajectory_rollout.py   # Multi-step rollout experiments
  grid_search.py          # Hyperparameter grid search utility
notebooks/                # Optional analysis & plots
```

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
python src/eval_ik_models.py \
  --csv-path data/kuka_traj_dataset_traj.csv \
  --use-orientation \
  --mlp-ckpt mlp_ik_traj_checkpoints/ikmlp-epoch=AAA-val_loss=BBB.ckpt \
  --gnn-ckpt gnn_ik_checkpoints/gnnik-epoch=XXX-val_loss=YYY.ckpt \
  --num-samples 200
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

**Note:** Classical IK baselines are automatically evaluated and included in the report results (Section 7). They are compared against MLP and GNN models in all plots and metrics.

To evaluate classical IK methods separately:

```bash
python -c "
from src.classical_ik import *
from src.data_utils import load_traj_csv
import numpy as np

# Setup
cid = connect_pybullet(gui=False)
kuka_uid = load_kuka()
joint_indices = get_revolute_joints(kuka_uid)
ee_link_index = joint_indices[-1]
joint_l, joint_u = get_joint_limits(kuka_uid, joint_indices)

# Load data
poses, q_prev, q_curr, pose_dim = load_traj_csv('data/kuka_traj_dataset_traj.csv', use_orientation=True)

# Example: Solve IK for first sample
target_pos = poses[0][:3]
target_orn = poses[0][3:7] if pose_dim == 7 else None
q_init = q_prev[0]

# DLS IK
q_dls, info = ik_damped_least_squares(
    kuka_uid, joint_indices, ee_link_index,
    target_pos, target_orn, q_init,
    joint_l, joint_u
)

# PyBullet IK
q_pb = ik_pybullet_builtin(
    kuka_uid, joint_indices, ee_link_index,
    target_pos, target_orn, joint_l, joint_u
)

print('DLS solution:', q_dls)
print('PyBullet solution:', q_pb)
print('DLS info:', info)
"
```

---

## 📊 7. Generate Report Results and Plots

Generate all results and plots for the report:

```bash
# Update checkpoint paths in run_report_generation.sh first, then:
./run_report_generation.sh

# Or run directly:
python src/generate_report_results.py \
  --csv-path data/kuka_traj_dataset_traj.csv \
  --mlp-ckpt mlp_ik_traj_checkpoints/ikmlp-epoch=AAA-val_loss=BBB.ckpt \
  --gnn-ckpt gnn_ik_checkpoints/gnnik-epoch=XXX-val_loss=YYY.ckpt \
  --results-dir results
```

This generates:
- **Trajectory rollout evaluation**: Stability and cumulative EE drift over long trajectories
  - Plot: EE error over trajectory steps (mean ± std)
  - Plot: Cumulative EE drift over trajectories
  - Data: Statistics (mean/std/max EE error, cumulative drift, mean Δq norm)
- **Lambda sweep results**: Accuracy vs smoothness tradeoff (requires models trained with different λ values)
  - Plot: Joint MSE vs Lambda
  - Plot: EE MSE vs Lambda
  - Plot: Mean Δq (smoothness) vs Lambda
  - Plot: Accuracy vs Smoothness tradeoff
  - Data: Metrics for each lambda value
- **All plots for report** (includes classical IK baselines):
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

