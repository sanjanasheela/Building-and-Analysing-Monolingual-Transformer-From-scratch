#!/usr/bin/env bash
# ==============================================================================
# run_bonus.sh — Run Bonus Ablation (No Positional Embeddings) on Kaggle / Local
# ==============================================================================
# Usage:
#   bash bonus/run_bonus.sh [MAX_STEPS] [DATA_DIR]
#
# Examples:
#   bash bonus/run_bonus.sh 15000
#   bash bonus/run_bonus.sh 15000 /kaggle/input/telugu-shards
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MAX_STEPS="${1:-15000}"
DATA_DIR="${2:-${PROJECT_DIR}/token_shards_7500}"

echo "========================================================================"
echo "Starting Bonus Ablation Study: Model H (Telugu) without Positional Embeddings"
echo "========================================================================"
echo "Project Directory : ${PROJECT_DIR}"
echo "Bonus Directory   : ${SCRIPT_DIR}"
echo "Target Max Steps  : ${MAX_STEPS}"
echo "Token Shards Dir  : ${DATA_DIR}"
echo "========================================================================"

# Activate virtual environment if available
if [ -f "/home/sanjana/Documents/7/LMA/.venv/bin/activate" ]; then
    echo "Activating virtual environment: /home/sanjana/Documents/7/LMA/.venv"
    source /home/sanjana/Documents/7/LMA/.venv/bin/activate
fi

# Print GPU info
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "nvidia-smi not found. Training on available device."
fi

# Step 1: Pretrain ablated model
echo ""
echo ">>> STEP 1: Retraining Ablated Model H for ${MAX_STEPS} steps..."
python "${SCRIPT_DIR}/train_ablation.py" \
    --config "${SCRIPT_DIR}/config.yaml" \
    --max_steps "${MAX_STEPS}" \
    --data_dir "${DATA_DIR}" \
    --output_dir "${SCRIPT_DIR}/checkpoints"

# Step 2: Run Full Phase 2 Evaluation Suite
echo ""
echo ">>> STEP 2: Running Full Phase 2 Evaluation Suite on Ablated Model..."
python "${SCRIPT_DIR}/eval_ablation.py" \
    --config "${SCRIPT_DIR}/config.yaml" \
    --report_dir "${PROJECT_DIR}/report/bonus"

echo ""
echo "========================================================================"
echo "✅ Bonus Ablation Training and Evaluation Completed!"
echo "All results, tables, heatmaps, and metrics are located in:"
echo "  ${PROJECT_DIR}/report/bonus"
echo "========================================================================"
