#!/usr/bin/env bash
# ==============================================================================
# 4 REPRESENTATIVE ALIGNMENT BASELINES BENCHMARK (5 EPOCHS x 3 LANGUAGES)
# ==============================================================================
# 1. Nhóm 1 (Attention Distillation) : align_to_distill (LREC-COLING 2024)
# 2. Nhóm 2 (Decoder Shifted State)  : shift_aet (EMNLP 2020)
# 3. Nhóm 3 (Embedding Alignment)    : awesome_align (EACL 2021 / CMU)
# 4. Nhóm 4 (Contrastive InfoNCE)    : cl_lsa (ACL 2024)
# ==============================================================================
# Parameters:
# - NUM_EPOCHS = 5
# - BATCH_SIZE = 16
# - LEARNING_RATE = 2e-5
# - MAX_SOURCE_LEN = 256
# - MAX_TARGET_LEN = 256
# - SEED = 42
# ==============================================================================

NUM_EPOCHS=5
BATCH_SIZE=16
LR=2e-5
MAX_LEN=256
SEED=42
OUTPUT_DIR="checkpoints"

LANGUAGES=("rhade" "tay" "bahnaric")
MODELS=(
    "align_to_distill"
    "shift_aet"
    "awesome_align"
    "cl_lsa"
)

SUCCESS_LIST=()
SKIPPED_LIST=()
FAILED_LIST=()

echo "========================================================================"
echo "    🚀 KHỞI ĐỘNG CHẠY 4 BASELINE ĐẠI DIỆN TRÊN 3 NGÔN NGỮ (5 EPOCHS)"
echo "========================================================================"
echo "[*] Các phương pháp: align_to_distill, shift_aet, awesome_align, cl_lsa"
echo "[*] Ngôn ngữ        : rhade, tay, bahnaric"

for LANG in "${LANGUAGES[@]}"; do
    echo ""
    echo "===================================================================="
    echo ">>> BẮT ĐẦU CHO NGÔN NGỮ: ${LANG^^}"
    echo "===================================================================="
    
    for MODEL in "${MODELS[@]}"; do
        CKPT_DIR="${OUTPUT_DIR}/${MODEL}_${LANG}"
        PRED_FILE="${CKPT_DIR}/test_predictions.csv"
        
        # Kiểm tra Smart Skip nếu đã có kết quả
        if [ -f "$PRED_FILE" ] && [ -s "$PRED_FILE" ]; then
            echo ""
            echo ">>> [⏭️ SMART SKIP] [${LANG^^}] ${MODEL} ĐÃ HOÀN TẤT trước đó. Bỏ qua!"
            SKIPPED_LIST+=("${MODEL}_${LANG}")
            continue
        fi

        echo ""
        echo ">>> [${LANG^^}] Đang huấn luyện: ${MODEL} ..."
        
        python train.py \
            --lang "${LANG}" \
            --model_type "${MODEL}" \
            --output_dir "${OUTPUT_DIR}" \
            --num_epochs "${NUM_EPOCHS}" \
            --batch_size "${BATCH_SIZE}" \
            --learning_rate "${LR}" \
            --max_source_length "${MAX_LEN}" \
            --max_target_length "${MAX_LEN}" \
            --seed "${SEED}" \
            --fp16
            
        EXIT_CODE=$?
        
        if [ $EXIT_CODE -eq 0 ] && [ -f "$PRED_FILE" ]; then
            echo ">>> [✅ THÀNH CÔNG] [${LANG^^}] Đã hoàn tất: ${MODEL}!"
            SUCCESS_LIST+=("${MODEL}_${LANG}")
        else
            echo ">>> [❌ LỖI] [${LANG^^}] Phương pháp ${MODEL} gặp lỗi (Exit code: ${EXIT_CODE})!"
            FAILED_LIST+=("${MODEL}_${LANG}")
        fi
    done
done

echo ""
echo "========================================================================"
echo "                    📊 TỔNG KẾT TIẾN TRÌNH 4 BASELINES"
echo "========================================================================"
echo "  [+] Đã hoàn tất từ trước : ${#SKIPPED_LIST[@]} mô hình"
echo "  [+] Huấn luyện mới xong  : ${#SUCCESS_LIST[@]} mô hình"
echo "  [!] Bị lỗi               : ${#FAILED_LIST[@]} mô hình"
echo ""
echo "💡 Chạy: python summary_results.py để xem toàn bộ bảng điểm SacreBLEU & chrF++!"
echo "========================================================================"
