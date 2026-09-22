#!/usr/bin/env bash
# ==============================================================================
# GIRA FAST SCREENING RUNNER (STRICTLY FAIR BENCHMARK SUITE)
# ==============================================================================
# Sàng lọc nhanh các cấu hình GIRA cốt lõi trên Tiếng Tày (BARTpho, Seed 42)
# Mốc đối chuẩn cứng (Hard Baselines to beat):
#   - Vanilla Baseline : 25.34 BLEU (chrF++ 36.05)
#   - AWESOME-align    : 25.39 BLEU (chrF++ 36.30)
#
# Danh sách 4 cấu hình:
#   1. A1 : gira_tay_seed42           (lambda=1.0, detach=True, learnable alpha)
#   2. A2 : gira_nodetach_tay_seed42  (lambda=1.0, detach=False -- Sanity Check)
#   3. A4a: gira_lam0.5_tay_seed42    (lambda=0.5, detach=True)
#   4. A4b: gira_lam2.0_tay_seed42    (lambda=2.0, detach=True)
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

echo "========================================================================"
echo "    🚀 KHỞI ĐỘNG SÀNG LỌC PHƯƠNG PHÁP GIRA TRÊN TIẾNG TÀY (SEED 42)"
echo "========================================================================"
echo "[*] Ngôn ngữ thực nghiệm : ${TARGET_LANG^^} -> VI"
echo "[*] Seed đối chuẩn       : ${TARGET_SEED}"
echo "[*] Thư mục lưu biệt lập : ${OUTPUT_DIR}"
echo "[*] Cấu hình huấn luyện  : Epochs=${NUM_EPOCHS}, Batch=${BATCH_SIZE}, LR=${LR_BARTPHO}, FP16=True"
echo "========================================================================"

trap 'echo -e "\n[!] Đã nhận tín hiệu hủy (Ctrl+C). Dừng an toàn..."; exit 1;' INT

# Định nghĩa các cấu hình thí nghiệm: EXP_NAME | LAMBDA | EXTRA_FLAGS
RUNS=(
    "gira_tay_seed42|1.0|--gira_layer -1"
    "gira_nodetach_tay_seed42|1.0|--gira_no_detach --gira_layer -1"
    "gira_lam0.5_tay_seed42|0.5|--gira_layer -1"
    "gira_lam2.0_tay_seed42|2.0|--gira_layer -1"
)

for ITEM in "${RUNS[@]}"; do
    IFS="|" read -r EXP_NAME LAMBDA EXTRA_FLAGS <<< "${ITEM}"
    CKPT_DIR="${OUTPUT_DIR}/${EXP_NAME}"

    if [ -f "${CKPT_DIR}/eval_metrics.json" ] && [ -f "${CKPT_DIR}/test_predictions.csv" ]; then
        echo ""
        echo ">>> [⏭️ SMART SKIP] ${EXP_NAME} đã hoàn thành trước đó. Bỏ qua!"
        continue
    fi

    echo ""
    echo "===================================================================="
    echo ">>> [🚀 RUNNING] ${EXP_NAME} (Lambda=${LAMBDA}, Flags=${EXTRA_FLAGS})"
    echo "===================================================================="

    START_TIME=$(date +%s)

    python train_fair_benchmark.py \
        --model_type "gira" \
        --model_ckpt "vinai/bartpho-syllable" \
        --lang "${TARGET_LANG}" \
        --output_dir "${OUTPUT_DIR}" \
        --exp_name "${EXP_NAME}" \
        --gira_lambda "${LAMBDA}" \
        ${EXTRA_FLAGS} \
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

echo ""
echo "========================================================================"
echo "    📊 TỔNG HỢP VÀ ĐỐI CHIẾU KẾT QUẢ GIRA VỚI CÁC BASELINE HIỆN CÓ"
echo "========================================================================"

python scripts/report_fair_benchmark.py "${OUTPUT_DIR}"

echo ""
echo "========================================================================"
echo " 🎯 CỔNG QUYẾT ĐỊNH (DECISION GATE CHECK):"
echo " - Nếu A1 (gira) >= 25.75 - 25.85 BLEU và A2 (no_detach) sập sâu: [GO] Đột phá hoàn hảo!"
echo " - Nếu A1 ~ 25.34 (ngang Vanilla) và A2 sập sâu: [EMPIRICAL FINDINGS] Bài báo Negative Result chuẩn mực!"
echo "========================================================================"
