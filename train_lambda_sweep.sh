#!/bin/bash
# Script to train MLP and GNN models with different lambda values for lambda sweep analysis
# Lambda values: 1e-3 (0.001), 1e-2 (0.01), 1e-1 (0.1), 1e0 (1.0)

set -e  # Exit on error

CSV_PATH="data/kuka_traj_dataset_traj.csv"
BASE_DIR=$(pwd)

# Lambda values (logarithmic scale)
LAMBDA_VALUES=(0.001 0.01 0.1 1.0)

echo "=========================================="
echo "Training Models for Lambda Sweep"
echo "=========================================="
echo ""
echo "Lambda values: ${LAMBDA_VALUES[@]}"
echo "CSV path: $CSV_PATH"
echo ""

# Check if CSV exists
if [ ! -f "$CSV_PATH" ]; then
    echo "Error: CSV file not found: $CSV_PATH"
    echo "Please generate the trajectory dataset first (see README Section 1)"
    exit 1
fi

# Create subdirectories for each lambda value
mkdir -p mlp_ik_traj_checkpoints
mkdir -p gnn_ik_checkpoints

# Train MLP models with different lambda values
echo "=========================================="
echo "Training MLP Models"
echo "=========================================="
for lambda in "${LAMBDA_VALUES[@]}"; do
    echo ""
    echo "Training MLP with λ=$lambda..."
    echo "----------------------------------------"
    
    # Create subdirectory for this lambda (optional, for organization)
    lambda_dir="mlp_ik_traj_checkpoints/lambda_${lambda}"
    mkdir -p "$lambda_dir"
    
    # Note: Checkpoints will be saved to mlp_ik_traj_checkpoints/ by default
    # The subdirectory is just for organization if you want to move them later
    python src/mlp_ik.py \
        --csv-path "$CSV_PATH" \
        --use-orientation \
        --traj-mode \
        --batch-size 256 \
        --max-epochs 100 \
        --hidden-dims 256 256 128 \
        --lr 1e-3 \
        --weight-decay 1e-4 \
        --dropout 0.1 \
        --lambda-movement "$lambda" \
        --accelerator auto
    
    echo "✓ MLP training with λ=$lambda completed"
done

echo ""
echo "=========================================="
echo "Training GNN Models"
echo "=========================================="
for lambda in "${LAMBDA_VALUES[@]}"; do
    echo ""
    echo "Training GNN with λ=$lambda..."
    echo "----------------------------------------"
    
    python src/gnn_ik.py \
        --csv-path "$CSV_PATH" \
        --use-orientation \
        --hidden-dim 64 \
        --num-layers 3 \
        --lambda-movement "$lambda" \
        --batch-size 128 \
        --max-epochs 100 \
        --accelerator auto
    
    echo "✓ GNN training with λ=$lambda completed"
done

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo ""
echo "Checkpoints saved to:"
echo "  - mlp_ik_traj_checkpoints/"
echo "  - gnn_ik_checkpoints/"
echo ""
echo "Next step: Run lambda sweep analysis"
echo "  python src/generate_report_results.py \\"
echo "    --csv-path $CSV_PATH \\"
echo "    --results-dir results"
echo ""
echo "The script will automatically find the best checkpoints for each lambda value!"
echo "No need to manually specify checkpoint paths."

