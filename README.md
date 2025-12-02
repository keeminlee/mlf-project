# KUKA iiwa Inverse Kinematics Learning Project  
**Machine Learning for Robotics — Final Project**

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
  --mlp-ckpt mlp_ik_traj_checkpoints/ikmlp-epoch=XXX-val_loss=YYY.ckpt \
  --gnn-ckpt gnn_ik_checkpoints/gnnik-epoch=AAA-val_loss=BBB.ckpt \
  --num-samples 200
```

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
- PyBullet’s built-in IK solver  

Useful for comparisons.

---

## 📝 Notes

- PyBullet runs in DIRECT (headless) mode  
- All models implemented in PyTorch Lightning  
- GNN uses PyTorch Geometric  
- Supports CPU / CUDA / MPS (Apple Silicon)
