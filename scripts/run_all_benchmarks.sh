#!/usr/bin/env bash
# ==============================================================================
# UNIFIED MAIN BENCHMARK RUNNER SCRIPT (TSSA & BASELINES)
# ==============================================================================
# Tất cả các thí nghiệm đều chạy theo đúng chuẩn siêu tham số đồng nhất:
# - NUM_EPOCHS = 10 (Early stopping patience = 3)
# - BATCH_SIZE = 16
# - LEARNING_RATE = 2e-5
# - WEIGHT_DECAY = 0.01
# - MAX_SOURCE_LEN = 256
# - MAX_TARGET_LEN = 256
# - SEED = 42
# ==============================================================================

set -e

NUM_EPOCHS=5
BATCH_SIZE=16
LR=2e-5
MAX_LEN=256
SEED=42

LANGUAGES=("rhade" "tay" "bahnaric")
MODELS=("bartpho_vanilla" "guided_attn" "joint_align" "awesome_align" "cl_lsa" "tssa")

echo "========================================================================"
echo "    BẮT ĐẦU CHẠY TOÀN BỘ BENCHMARK CHÍNH (3 NGÔN NGỮ x 6 MÔ HÌNH)"
echo "========================================================================"

for LANG in "${LANGUAGES[@]}"; do
    echo ""
    echo "===================================================================="
    echo ">>> BẮT ĐẦU CHẠY CHO NGÔN NGỮ: ${LANG^^}"
    echo "===================================================================="
    
    for MODEL in "${MODELS[@]}"; do
        echo ""
        echo ">>> [${LANG^^}] Đang chạy mô hình: ${MODEL} ..."
        python train.py \
            --lang "${LANG}" \
            --model_type "${MODEL}" \
            --num_epochs "${NUM_EPOCHS}" \
            --batch_size "${BATCH_SIZE}" \
            --learning_rate "${LR}" \
            --max_source_length "${MAX_LEN}" \
            --max_target_length "${MAX_LEN}" \
            --seed "${SEED}" \
            --fp16
            
        echo ">>> [${LANG^^}] Hoàn tất: ${MODEL}!"
    done
done

echo ""
echo "========================================================================"
echo "    CHÚC MỪNG! TOÀN BỘ THÍ NGHIỆM MAIN BENCHMARK ĐÃ HOÀN TẤT!"
echo "========================================================================"
