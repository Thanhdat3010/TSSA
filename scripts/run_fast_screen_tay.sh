#!/usr/bin/env bash
# ==============================================================================
# TSSA-PRO RAPID SCREENING SCRIPT (TÀY: BARTPHO & VIT5)
# ==============================================================================
# Fast screening on the most isomorphic, highest-signal language pair (Tày -> Việt)
# Usage:
#   bash scripts/run_fast_screen_tay.sh [LAMBDA_STRUCT]
# Example:
#   bash scripts/run_fast_screen_tay.sh 1.0
# ==============================================================================

set -e

NUM_EPOCHS=5
BATCH_SIZE=16
MAX_LEN=256
SEED=42
OUTPUT_DIR="checkpoints/tssa_pro"

LR_BARTPHO=2e-5
LR_VIT5=1e-4

PRIME_TAU=0.07
CONF_THRESHOLD=0.20
ENTROPY_TAU=0.50

LAMBDA_STRUCT="${1:-${LAMBDA_STRUCT:-1.00}}"
LAMBDA_PRIME=0.08
LAMBDA_ROUTE=0.00
SIGMA_KAPPA=0.75

mkdir -p "${OUTPUT_DIR}"

echo "========================================================================"
echo "    ⚡ TSSA-PRO RAPID SCREENING: TÀY -> TIẾNG VIỆT"
echo "========================================================================"
echo "[*] Thử nghiệm nhanh trên cặp tín hiệu mạnh nhất (Tày):"
echo "[*] LAMBDA_STRUCT = ${LAMBDA_STRUCT} (Sweep candidate)"
echo "[*] BARTpho LR    = ${LR_BARTPHO} (Fair standard)"
echo "[*] ViT5 LR       = ${LR_VIT5} (Fair standard)"
echo "========================================================================"

# 1. BARTpho Tày
echo ""
echo ">>> [1/2] Huấn luyện BARTpho Tày..."
python train_tssa_pro.py \
    --lang "tay" \
    --model_ckpt "vinai/bartpho-syllable" \
    --exp_name "tssa_tay" \
    --output_dir "${OUTPUT_DIR}" \
    --num_epochs "${NUM_EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --learning_rate "${LR_BARTPHO}" \
    --max_source_length "${MAX_LEN}" \
    --max_target_length "${MAX_LEN}" \
    --seed "${SEED}" \
    --fp16 \
    --sigma_kappa "${SIGMA_KAPPA}" \
    --use_struct \
    --use_prime \
    --no_route \
    --use_centering \
    --protect_struct_fertility \
    --entropy_tau "${ENTROPY_TAU}" \
    --lambda_struct "${LAMBDA_STRUCT}" \
    --lambda_prime "${LAMBDA_PRIME}" \
    --lambda_route "${LAMBDA_ROUTE}" \
    --prime_tau "${PRIME_TAU}" \
    --conf_threshold "${CONF_THRESHOLD}"

# 2. ViT5 Tày
echo ""
echo ">>> [2/2] Huấn luyện ViT5 Tày..."
python train_tssa_pro.py \
    --lang "tay" \
    --model_ckpt "VietAI/vit5-base" \
    --exp_name "vit5_tssa_tay" \
    --output_dir "${OUTPUT_DIR}" \
    --num_epochs "${NUM_EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --learning_rate "${LR_VIT5}" \
    --max_source_length "${MAX_LEN}" \
    --max_target_length "${MAX_LEN}" \
    --seed "${SEED}" \
    --fp16 \
    --sigma_kappa "${SIGMA_KAPPA}" \
    --use_struct \
    --use_prime \
    --no_route \
    --use_centering \
    --protect_struct_fertility \
    --entropy_tau "${ENTROPY_TAU}" \
    --lambda_struct "${LAMBDA_STRUCT}" \
    --lambda_prime "${LAMBDA_PRIME}" \
    --lambda_route "${LAMBDA_ROUTE}" \
    --prime_tau "${PRIME_TAU}" \
    --conf_threshold "${CONF_THRESHOLD}"

# 3. So sánh nhanh
echo ""
echo "========================================================================"
echo ">>> KẾT QUẢ ĐỐI SOÁT VỚI BASELINE (TÀY):"
echo "========================================================================"
python compare_tssa_pro_vs_baselines.py
