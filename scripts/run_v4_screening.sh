#!/usr/bin/env bash
# ==============================================================================
# TSSA-V4 FAST SCREENING RUNNER (DECOUPLED MIDDLE-LAYER SEMANTIC ANCHORING)
# ==============================================================================
# Sàng lọc nhanh 3 biến thể kiến trúc TSSA-V4 trên Tiếng Tày (Seed 42)
# Mốc đối chuẩn cứng (Hard Baselines to beat từ checkpoints/fair_benchmark):
#   - Vanilla Baseline : 25.34 BLEU (chrF++ 36.05)
#   - AWESOME-align    : 25.39 BLEU (chrF++ 36.30)
#   - TSSA-Pro (Cũ)    : 25.36 BLEU (chrF++ 36.00)
#
# 3 Biến thể sàng lọc:
#   1. v4_sent   : InfoNCE Mức Câu qua Decoupled MLP Projector + Memory Queue 256
#   2. v4_tok    : Soft Barycenter Mức Token qua Decoupled MLP Projector
#   3. v4_hybrid : Kết hợp cả Câu (InfoNCE) + Token (Barycenter)
#
# Cổng Quyết Định (Pre-registered Decision Gate):
#   - GO   : Nếu có biến thể đạt >= 25.75 - 25.85 BLEU (Δ >= +0.40 ~ +0.50) -> Tiếp tục mở rộng.
#   - STOP : Nếu cả 3 biến thể kẹt ở ~25.3 - 25.4 BLEU -> Khẳng định căn chỉnh biểu diễn trên BARTpho bão hòa.
#
# Usage:
#   bash scripts/run_v4_screening.sh [tay] [42]
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

V4_METHODS=(
    "v4_sent"
    "v4_tok"
    "v4_hybrid"
)

echo "========================================================================"
echo "    🚀 KHỞI ĐỘNG SÀNG LỌC NHANH KIẾN TRÚC TSSA-V4 (MIDDLE-LAYER)"
echo "========================================================================"
echo "[*] Ngôn ngữ thực nghiệm : ${TARGET_LANG^^} -> VI"
echo "[*] Seed thực nghiệm     : ${TARGET_SEED}"
echo "[*] Thư mục lưu biệt lập : ${OUTPUT_DIR}"
echo "[*] Danh sách biến thể   : ${V4_METHODS[*]}"
echo "[*] Cấu hình huấn luyện  : Epochs=${NUM_EPOCHS}, Batch=${BATCH_SIZE}, LR=${LR_BARTPHO}, FP16=True"
echo "========================================================================"

trap 'echo -e "\n[!] Đã nhận tín hiệu hủy (Ctrl+C). Dừng an toàn..."; exit 1;' INT

for METHOD in "${V4_METHODS[@]}"; do
    EXP_NAME="${METHOD}_${TARGET_LANG}_seed${TARGET_SEED}"
    CKPT_DIR="${OUTPUT_DIR}/${EXP_NAME}"

    # ----------------------------------------------------------------------
    # CƠ CHẾ SMART SKIP: NẾU ĐÃ CÓ KẾT QUẢ THÌ BỎ QUA
    # ----------------------------------------------------------------------
    if [ -f "${CKPT_DIR}/eval_metrics.json" ] && [ -f "${CKPT_DIR}/test_predictions.csv" ]; then
        echo ""
        echo ">>> [⏭️ SMART SKIP] ${EXP_NAME} đã hoàn thành trước đó. Bỏ qua!"
        continue
    fi

    echo ""
    echo "===================================================================="
    echo ">>> [🚀 RUNNING] ${EXP_NAME} (Method=${METHOD}, Seed=${TARGET_SEED})"
    echo "===================================================================="

    START_TIME=$(date +%s)

    python train_fair_benchmark.py \
        --model_type "${METHOD}" \
        --model_ckpt "vinai/bartpho-syllable" \
        --lang "${TARGET_LANG}" \
        --output_dir "${OUTPUT_DIR}" \
        --exp_name "${EXP_NAME}" \
        --num_epochs "${NUM_EPOCHS}" \
        --batch_size "${BATCH_SIZE}" \
        --learning_rate "${LR_BARTPHO}" \
        --max_source_length "${MAX_LEN}" \
        --max_target_length "${MAX_LEN}" \
        --seed "${TARGET_SEED}" \
        --v4_queue_size 256 \
        --fp16

    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    echo ">>> [✓ HOÀN TẤT] ${EXP_NAME} sau ${ELAPSED} giây!"
done

echo ""
echo "========================================================================"
echo "    📊 TỔNG HỢP VÀ ĐỐI CHIẾU KẾT QUẢ V4 VỚI CÁC BASELINE HIỆN CÓ"
echo "========================================================================"

python scripts/report_fair_benchmark.py "${OUTPUT_DIR}"

echo ""
echo "========================================================================"
echo " 🎯 CỔNG QUYẾT ĐỊNH (DECISION GATE CHECK):"
echo " - Nếu bất kỳ V4 nào đạt >= 25.75 - 25.85 BLEU: [GO] Vượt trội rõ rệt Vanilla (25.34)!"
echo " - Nếu cả 3 V4 dao động ~ 25.30 - 25.40 BLEU  : [STOP] Căn chỉnh biểu diễn trên BARTpho đã bão hòa."
echo "========================================================================"
