#!/bin/bash
# Script to generate all report results and plots

# Set checkpoint paths (update these with your actual checkpoint files)
MLP_CKPT="mlp_ik_traj_checkpoints/ikmlp-epoch=012-val_loss=0.0025.ckpt"
GNN_CKPT="gnn_ik_checkpoints/gnnik-epoch=019-val_loss=0.0025.ckpt"
CSV_PATH="data/kuka_traj_dataset_traj.csv"

echo "=========================================="
echo "Generating Report Results and Plots"
echo "=========================================="
echo ""
echo "Checkpoints:"
echo "  MLP: $MLP_CKPT"
echo "  GNN: $GNN_CKPT"
echo "  CSV: $CSV_PATH"
echo ""

# Run the report generation script
python src/generate_report_results.py \
  --csv-path "$CSV_PATH" \
  --mlp-ckpt "$MLP_CKPT" \
  --gnn-ckpt "$GNN_CKPT" \
  --results-dir results

echo ""
echo "=========================================="
echo "Done! Check results/ directory for:"
echo "  - results/plots/ : All generated plots"
echo "  - results/data/  : All data files (JSON, NPZ)"
echo "=========================================="

