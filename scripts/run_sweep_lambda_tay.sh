#!/usr/bin/env bash
# ==============================================================================
# TSSA-PRO: LAMBDA-SWEEP RAPID SCREENING TRÊN TIẾNG TÀY (ViT5 & BARTpho)
# ==============================================================================
# Mục đích: Quét nhanh dải trọng số lambda_struct in {0.2, 1.0, 5.0} trên tiếng Tày
# để tìm ra "điểm ngọt" tối ưu nhất cho Cosine Hypersphere trước khi chạy toàn bộ.
#
# Cách dùng:
#   bash scripts/run_sweep_lambda_tay.sh [BACKBONE]
# Ví dụ:
#   bash scripts/run_sweep_lambda_tay.sh vit5       (Quét trên ViT5 - Khuyên dùng)
#   bash scripts/run_sweep_lambda_tay.sh bartpho    (Quét trên BARTpho)
#   bash scripts/run_sweep_lambda_tay.sh both       (Quét trên cả 2)
# ==============================================================================

set -e

BACKBONE="${1:-vit5}"
NUM_EPOCHS=5
BATCH_SIZE=16
MAX_LEN=256
SEED=42
OUTPUT_DIR="checkpoints/tssa_sweep_tay"

mkdir -p "${OUTPUT_DIR}"

LAMBDA_VALUES=(0.2 1.0 5.0)

echo "========================================================================"
echo "    🔍 TSSA-PRO LAMBDA-SWEEP SCREENING TRÊN TIẾNG TÀY (SEED 42)"
echo "========================================================================"
echo "[*] Mục tiêu       : Tìm lambda_struct tối ưu trong dải: ${LAMBDA_VALUES[*]}"
echo "[*] Backbone chọn  : ${BACKBONE^^}"
echo "[*] Cố định        : Centering=True, Normalized Entropy tau_H=0.50, FP16"
echo "[*] Thư mục lưu    : ${OUTPUT_DIR}"
echo "========================================================================"

run_sweep_for_model() {
    local MODEL_TYPE="$1" # "vit5" or "bartpho"
    local MODEL_CKPT="$2"
    local LR="$3"
    local VANILLA_BLEU="$4"
    local SOTA_BLEU="$5"

    echo ""
    echo "========================================================================"
    echo ">>> BẮT ĐẦU SWEEP CHO ${MODEL_TYPE^^} (Vanilla BLEU = ${VANILLA_BLEU})"
    echo "========================================================================"

    for LAMBDA in "${LAMBDA_VALUES[@]}"; do
        local EXP_NAME="${MODEL_TYPE}_tay_lambda_${LAMBDA}"
        local SAVE_PATH="${OUTPUT_DIR}/${EXP_NAME}"

        echo ""
        echo ">>> [${MODEL_TYPE^^} - TÀY] Huấn luyện với LAMBDA_STRUCT = ${LAMBDA}..."
        
        START_TIME=$(date +%s)

        python train_tssa_pro.py \
            --lang "tay" \
            --model_ckpt "${MODEL_CKPT}" \
            --exp_name "${EXP_NAME}" \
            --output_dir "${OUTPUT_DIR}" \
            --num_epochs "${NUM_EPOCHS}" \
            --batch_size "${BATCH_SIZE}" \
            --learning_rate "${LR}" \
            --max_source_length "${MAX_LEN}" \
            --max_target_length "${MAX_LEN}" \
            --seed "${SEED}" \
            --fp16 \
            --sigma_kappa "0.75" \
            --use_struct \
            --use_prime \
            --no_route \
            --use_centering \
            --protect_struct_fertility \
            --entropy_tau "0.50" \
            --lambda_struct "${LAMBDA}" \
            --lambda_prime "0.08" \
            --lambda_route "0.00" \
            --prime_tau "0.07" \
            --conf_threshold "0.20"

        END_TIME=$(date +%s)
        ELAPSED=$((END_TIME - START_TIME))
        echo "[✓] Xong lambda=${LAMBDA} sau ${ELAPSED}s."
    done

    # Bảng tổng kết kết quả sweep
    python scripts/report_sweep.py "${OUTPUT_DIR}"
}

if [ "${BACKBONE}" == "vit5" ] || [ "${BACKBONE}" == "both" ]; then
    run_sweep_for_model "vit5" "VietAI/vit5-base" "1e-4" "34.99" "35.83"
fi

if [ "${BACKBONE}" == "bartpho" ] || [ "${BACKBONE}" == "both" ]; then
    run_sweep_for_model "bartpho" "vinai/bartpho-syllable" "2e-5" "24.67" "25.20"
fi

echo ""
echo "🎉 [HOÀN TẤT SWEEP] Hãy chọn giá trị lambda có BLEU cao nhất để chạy cho các ngôn ngữ còn lại!"
