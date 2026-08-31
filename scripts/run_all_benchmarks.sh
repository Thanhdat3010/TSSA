#!/usr/bin/env bash
# ==============================================================================
# UNIFIED MAIN BENCHMARK RUNNER SCRIPT (TSSA 2.0 & ALL 8 ALIGNMENT BASELINES)
# ==============================================================================
# Unified training configuration across all methods & languages:
# - NUM_EPOCHS = 5
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
MODELS=(
    "align_to_distill"
    "structural_supervision"
    "shift_aet"
    "cross_init"
    "awesome_align"
    "dm_bli"
    "cl_lsa"
    "dpo_align"
)

echo "========================================================================"
echo "    BẮT ĐẦU CHẠY TOÀN BỘ 8 BASELINE BENCHMARK TRÊN 3 NGÔN NGỮ"
echo "========================================================================"

for LANG in "${LANGUAGES[@]}"; do
    echo ""
    echo "===================================================================="
    echo ">>> BẮT ĐẦU CHẠY CHO NGÔN NGỮ: ${LANG^^}"
    echo "===================================================================="
    
    for MODEL in "${MODELS[@]}"; do
        echo ""
        echo ">>> [${LANG^^}] Đang huấn luyện phương pháp: ${MODEL} ..."
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
echo "    CHÚC MỪNG! TOÀN BỘ 8 BASELINE BENCHMARK ĐÃ HOÀN TẤT THÀNH CÔNG!"
echo "========================================================================"
