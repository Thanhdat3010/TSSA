#!/usr/bin/env bash
# ==============================================================================
# 10-EPOCH BENCHMARK RUNNER (VANILLA BARTPHO vs TSSA 2.0 ACROSS 3 LANGUAGES)
# ==============================================================================
# Saves to checkpoints_10epochs/ (100% preserving existing 5-epoch checkpoints)
# - NUM_EPOCHS = 10 (Early stopping patience = 3)
# - BATCH_SIZE = 16
# - LEARNING_RATE = 2e-5
# - MAX_SOURCE_LEN = 256
# - MAX_TARGET_LEN = 256
# - SEED = 42
# ==============================================================================

NUM_EPOCHS=10
BATCH_SIZE=16
LR=2e-5
MAX_LEN=256
SEED=42
OUTPUT_DIR="checkpoints_10epochs"

LANGUAGES=("rhade" "tay" "bahnaric")
MODELS=("bartpho_vanilla" "tssa")

mkdir -p "${OUTPUT_DIR}"

echo "========================================================================"
echo "    🚀 KHỞI ĐỘNG HUẤN LUYỆN 10 EPOCHS: VANILLA BARTPHO vs TSSA 2.0"
echo "========================================================================"
echo "[*] Lưu trữ độc lập tại: ${OUTPUT_DIR}/ (Không ảnh hưởng checkpoints/ 5 epochs)"

for LANG in "${LANGUAGES[@]}"; do
    echo ""
    echo "===================================================================="
    echo ">>> BẮT ĐẦU CHO NGÔN NGỮ: ${LANG^^}"
    echo "===================================================================="
    
    for MODEL in "${MODELS[@]}"; do
        CKPT_DIR="${OUTPUT_DIR}/${MODEL}_${LANG}"
        PRED_FILE="${CKPT_DIR}/test_predictions.csv"
        
        # Kiểm tra Smart Skip nếu mô hình 10 epochs này đã hoàn tất
        if [ -f "$PRED_FILE" ] && [ -s "$PRED_FILE" ]; then
            echo ""
            echo ">>> [⏭️ SMART SKIP] [${LANG^^}] ${MODEL} ĐÃ HOÀN TẤT 10 EPOCHS. Bỏ qua!"
            continue
        fi

        echo ""
        echo ">>> [${LANG^^}] Đang huấn luyện 10 Epochs: ${MODEL} ..."
        
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
            
        echo ">>> [✅ HOÀN TẤT] [${LANG^^}] ${MODEL} (10 Epochs)!"
    done
done

echo ""
echo "========================================================================"
echo "    🎉 CHÚC MỪNG! TOÀN BỘ 6 MÔ HÌNH 10 EPOCHS ĐÃ HOÀN TẤT THÀNH CÔNG!"
echo "========================================================================"
echo "💡 Chạy: python compare_5_vs_10_epochs.py để xem bảng so sánh tăng trưởng!"
echo "========================================================================"
