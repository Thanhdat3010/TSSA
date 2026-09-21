#!/usr/bin/env bash
# ==============================================================================
# FAIR BENCHMARK BATCH RUNNER (STRICTLY SYMMETRIC & FAIR BENCHMARK)
# ==============================================================================
# Executes Vanilla, Baselines (AWESOME, CL-LSA, A2D, Shift-AET), and TSSA-Pro
# using the EXACT same unified training script: train_fair_benchmark.py
#
# Usage:
#   bash scripts/run_fair_benchmark.sh [tay | rhade | bahnaric | all] [seed]
#
# Examples:
#   bash scripts/run_fair_benchmark.sh tay 42      # Chạy 6 mô hình trên Tiếng Tày, Seed 42
#   bash scripts/run_fair_benchmark.sh all 42      # Chạy 6 mô hình trên cả 3 ngôn ngữ
# ==============================================================================

set -e

TARGET_LANG="${1:-tay}"
TARGET_SEED="${2:-42}"
OUTPUT_DIR="checkpoints/fair_benchmark"

LR_BARTPHO=2e-5
MAX_LEN=256
BATCH_SIZE=16
NUM_EPOCHS=5

mkdir -p "${OUTPUT_DIR}"

if [ "${TARGET_LANG}" == "all" ]; then
    LANGUAGES=("tay" "rhade" "bahnaric")
else
    LANGUAGES=("${TARGET_LANG}")
fi

METHODS=(
    "vanilla"
    "awesome_align"
    "cl_lsa"
    "align_to_distill"
    "shift_aet"
    "tssa_pro"
)

echo "========================================================================"
echo "    🔬 KHỞI ĐỘNG HỆ THỐNG ĐỐI CHUẨN CÔNG BẰNG (FAIR BENCHMARK SUITE)"
echo "========================================================================"
echo "[*] Danh sách ngôn ngữ : ${LANGUAGES[*]}"
echo "[*] Danh sách phương pháp: ${METHODS[*]}"
echo "[*] Seed thực nghiệm   : ${TARGET_SEED}"
echo "[*] Thư mục lưu biệt lập: ${OUTPUT_DIR}"
echo "[*] Siêu tham số chuẩn : Epochs=${NUM_EPOCHS}, Batch=${BATCH_SIZE}, LR=${LR_BARTPHO}"
echo "========================================================================"

trap 'echo -e "\n[!] Đã nhận tín hiệu hủy (Ctrl+C). Dừng an toàn..."; exit 1;' INT

for LANG in "${LANGUAGES[@]}"; do
    echo ""
    echo "===================================================================="
    echo ">>> BẮT ĐẦU ĐỐI CHUẨN CHO NGÔN NGỮ: ${LANG^^}"
    echo "===================================================================="

    for METHOD in "${METHODS[@]}"; do
        EXP_NAME="${METHOD}_${LANG}_seed${TARGET_SEED}"
        CKPT_DIR="${OUTPUT_DIR}/${EXP_NAME}"

        # ----------------------------------------------------------------------
        # CƠ CHẾ SMART SKIP: NẾU ĐÃ HOÀN TẤT THÌ BỎ QUA ĐỂ TIẾT KIỆM GPU
        # ----------------------------------------------------------------------
        if [ -f "${CKPT_DIR}/eval_metrics.json" ] && [ -f "${CKPT_DIR}/test_predictions.csv" ]; then
            echo ""
            echo ">>> [⏭️ SMART SKIP] [${LANG^^}] ${EXP_NAME} đã có kết quả. Bỏ qua!"
            continue
        fi

        echo ""
        echo "===================================================================="
        echo ">>> [🚀 RUNNING] ${EXP_NAME} (Lang=${LANG}, Method=${METHOD}, Seed=${TARGET_SEED})"
        echo "===================================================================="

        START_TIME=$(date +%s)

        python train_fair_benchmark.py \
            --model_type "${METHOD}" \
            --model_ckpt "vinai/bartpho-syllable" \
            --lang "${LANG}" \
            --output_dir "${OUTPUT_DIR}" \
            --exp_name "${EXP_NAME}" \
            --num_epochs "${NUM_EPOCHS}" \
            --batch_size "${BATCH_SIZE}" \
            --learning_rate "${LR_BARTPHO}" \
            --max_source_length "${MAX_LEN}" \
            --max_target_length "${MAX_LEN}" \
            --seed "${TARGET_SEED}" \
            --fp16

        END_TIME=$(date +%s)
        ELAPSED=$((END_TIME - START_TIME))
        echo ">>> [✓ HOÀN TẤT] ${EXP_NAME} sau ${ELAPSED} giây!"
    done
done

echo ""
echo "========================================================================"
echo "    📊 TỔNG HỢP VÀ XUẤT BÁO CÁO ĐỐI CHUẨN SIÊU TỐC (<1 GIÂY)"
echo "========================================================================"

python scripts/report_fair_benchmark.py "${OUTPUT_DIR}"

echo ""
echo "🎉 [HOÀN TẤT] Bộ thực nghiệm Fair Benchmark đã hoàn thành thành công!"
