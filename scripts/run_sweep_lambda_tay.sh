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
    echo ""
    echo "========================================================================"
    echo "📊 BẢNG TỔNG HỢP KẾT QUẢ SWEEP ${MODEL_TYPE^^} (TÀY -> VIỆT):"
    echo "------------------------------------------------------------------------"
    printf "%-12s | %-10s | %-12s | %-15s | %-15s\n" "Lambda" "BLEU" "chrF++" "vs Vanilla (${VANILLA_BLEU})" "vs SOTA (${SOTA_BLEU})"
    echo "------------------------------------------------------------------------"
    for LAMBDA in "${LAMBDA_VALUES[@]}"; do
        local METRICS_FILE="${OUTPUT_DIR}/${MODEL_TYPE}_tay_lambda_${LAMBDA}/eval_metrics.json"
        if [ -f "${METRICS_FILE}" ]; then
            python -c "
import json
with open('${METRICS_FILE}') as f:
    d = json.load(f)
bleu = d.get('bleu', d.get('sacrebleu', 0.0))
chrf = d.get('chrf', d.get('chrf++', 0.0))
vanilla = float('${VANILLA_BLEU}')
sota = float('${SOTA_BLEU}')
diff_v = bleu - vanilla
diff_s = bleu - sota
sign_v = '+' if diff_v >= 0 else ''
sign_s = '+' if diff_s >= 0 else ''
print(f'lambda={LAMBDA:<5} | {bleu:<10.2f} | {chrf:<12.2f} | {sign_v}{diff_v:<14.2f} | {sign_s}{diff_s:<14.2f}')
"
        else
            printf "%-12s | %-10s | %-12s | %-15s | %-15s\n" "lambda=${LAMBDA}" "N/A" "N/A" "N/A" "N/A"
        fi
    done
    echo "========================================================================"
}

if [ "${BACKBONE}" == "vit5" ] || [ "${BACKBONE}" == "both" ]; then
    run_sweep_for_model "vit5" "VietAI/vit5-base" "1e-4" "34.99" "35.83"
fi

if [ "${BACKBONE}" == "bartpho" ] || [ "${BACKBONE}" == "both" ]; then
    run_sweep_for_model "bartpho" "vinai/bartpho-syllable" "2e-5" "24.67" "25.20"
fi

echo ""
echo "🎉 [HOÀN TẤT SWEEP] Hãy chọn giá trị lambda có BLEU cao nhất để chạy cho các ngôn ngữ còn lại!"
