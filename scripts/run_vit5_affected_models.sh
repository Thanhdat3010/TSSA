#!/usr/bin/env bash
# ==============================================================================
# TARGETED RERUN: ViT5 AFFECTED MODELS (TSSA & A2D ACROSS 3 LANGUAGES)
# ==============================================================================
# Re-runs ONLY the 2 methods affected by the native T5 Cross-Attention update:
#   1. vit5_align_to_distill_{lang}  : Cross-Attention Distillation (A2D, LREC-COLING 2024)
#   2. vit5_tssa_{lang}              : Target-Supervised Semantic Anchoring (Ours)
#
# Preserves 100% of unaffected baselines: Vanilla, Shift-AET, AWESOME, CL-LSA.
# Total: 2 methods x 3 languages = 6 models (~19-20 hours on GPU)
# ==============================================================================

set -e

BACKBONE="VietAI/vit5-base"
NUM_EPOCHS=5
BATCH_SIZE=16
LR=1e-4
MAX_LEN=256
SEED=42
OUTPUT_DIR="checkpoints"

LANGUAGES=("rhade" "tay" "bahnaric")
MODELS=("align_to_distill" "tssa")

SUCCESS_LIST=()
FAILED_LIST=()

echo "========================================================================"
echo "    🚀 KHỞI ĐỘNG CHẠY LẠI CÁC MÔ HÌNH ViT5 BỊ ẢNH HƯỞNG (TSSA & A2D)"
echo "========================================================================"
echo "[*] Backbone Model : ${BACKBONE}"
echo "[*] Các phương pháp: ${MODELS[*]}"
echo "[*] Các ngôn ngữ   : ${LANGUAGES[*]}"
echo "[*] Số mô hình     : $((${#MODELS[@]} * ${#LANGUAGES[@]})) models (5 epochs, batch_size=${BATCH_SIZE}, lr=${LR})"
echo "[*] Trọng số TSSA  : lambda_struct=0.5, lambda_prime=0.2, lambda_route=0.1 (Đồng nhất 100% BARTpho)"
echo "========================================================================"

trap 'echo -e "\n[!] Đã nhận tín hiệu dừng (Ctrl+C). Đang thoát..."; exit 1;' INT

for LANG in "${LANGUAGES[@]}"; do
    echo ""
    echo "===================================================================="
    echo ">>> BẮT ĐẦU CHẠY LẠI CHO NGÔN NGỮ: ${LANG^^}"
    echo "===================================================================="
    
    for MODEL in "${MODELS[@]}"; do
        EXP_NAME="vit5_${MODEL}_${LANG}"
        CKPT_DIR="${OUTPUT_DIR}/${EXP_NAME}"
        PRED_FILE="${CKPT_DIR}/test_predictions.csv"
        
        # Xóa bản dịch và checkpoint cũ để buộc mô hình huấn luyện lại với ma trận Attention chuẩn mới
        if [ -d "$CKPT_DIR" ]; then
            echo "[*] Dọn dẹp checkpoint cũ của ${EXP_NAME} để huấn luyện mới..."
            rm -rf "$CKPT_DIR"
        fi

        echo ""
        echo ">>> [${LANG^^}] Đang huấn luyện lại: ${EXP_NAME} (Backbone: ${BACKBONE}) ..."
        START_TIME=$(date +%s)
        
        EXTRA_FLAGS=""
        if [ "$MODEL" == "tssa" ]; then
            EXTRA_FLAGS="--use_struct --use_prime --use_route --lambda_struct 0.5 --lambda_prime 0.2 --lambda_route 0.1"
        fi

        set +e
        python train.py \
            --lang "${LANG}" \
            --model_ckpt "${BACKBONE}" \
            --model_type "${MODEL}" \
            --exp_name "${EXP_NAME}" \
            --output_dir "${OUTPUT_DIR}" \
            --num_epochs "${NUM_EPOCHS}" \
            --batch_size "${BATCH_SIZE}" \
            --learning_rate "${LR}" \
            --max_source_length "${MAX_LEN}" \
            --max_target_length "${MAX_LEN}" \
            --seed "${SEED}" \
            --fp16 \
            ${EXTRA_FLAGS}
            
        EXIT_CODE=$?
        set -e
        
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        
        if [ $EXIT_CODE -eq 0 ] && [ -f "$PRED_FILE" ]; then
            echo ">>> [✅ THÀNH CÔNG] [${LANG^^}] Đã hoàn tất ${EXP_NAME} trong ${DURATION}s!"
            SUCCESS_LIST+=("${EXP_NAME}")
        else
            echo ">>> [❌ LỖI] [${LANG^^}] Mô hình ${EXP_NAME} gặp lỗi (Exit code: ${EXIT_CODE})!"
            FAILED_LIST+=("${EXP_NAME}")
        fi
    done
done

echo ""
echo "========================================================================"
echo "                    📊 TỔNG KẾT TIẾN TRÌNH RERUN"
echo "========================================================================"
echo "  [+] Huấn luyện thành công : ${#SUCCESS_LIST[@]} / 6 mô hình"
echo "  [!] Bị lỗi                : ${#FAILED_LIST[@]} mô hình"
echo "========================================================================"

if [ ${#FAILED_LIST[@]} -eq 0 ]; then
    echo "🎉 CHÚC MỪNG: Toàn bộ 6 mô hình đã hoàn tất thành công!"
    echo "💡 Đang tự động chạy tổng kết kết quả và kiểm định ý nghĩa thống kê..."
    python summary_vit5_results.py --comet
    python eval_significance_vit5.py
else
    echo "⚠️ Chú ý: Một số mô hình gặp lỗi: ${FAILED_LIST[*]}"
fi
echo "========================================================================"
