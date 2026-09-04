#!/usr/bin/env bash
# ==============================================================================
# FULL BENCHMARK SUITE: ViT5 (VietAI/vit5-base) CROSS-ARCHITECTURE GENERALIZATION
# ==============================================================================
# Evaluates 6 methods across 3 Low-Resource Mon-Khmer / Tai-Kadai languages:
#   1. vit5_vanilla_{lang}           : Standard Seq2Seq NMT without semantic alignment
#   2. vit5_align_to_distill_{lang}  : Cross-Attention Distillation (A2D, LREC-COLING 2024)
#   3. vit5_shift_aet_{lang}         : Shifted Decoder Embedding Alignment (Shift-AET, EMNLP 2020)
#   4. vit5_awesome_align_{lang}     : Fine-tuning Alignment Objectives (AWESOME-align, EACL 2021)
#   5. vit5_cl_lsa_{lang}            : Cross-Lingual Contrastive InfoNCE (CL-LSA, ACL 2024)
#   6. vit5_tssa_{lang}              : Proposed Target-Supervised Semantic Anchoring (Ours)
#
# Languages: Rhade (Ê Đê), Tay (Tày), Bahnaric (Ba Na)
# Total: 6 methods x 3 languages = 18 models
# ==============================================================================

set -e # Exit immediately if a pipeline command fails unexpectedly

BACKBONE="VietAI/vit5-base"
NUM_EPOCHS=5
BATCH_SIZE=16
LR=1e-4
MAX_LEN=256
SEED=42
OUTPUT_DIR="checkpoints"

LANGUAGES=("rhade" "tay" "bahnaric")
MODELS=(
    "vanilla"
    "align_to_distill"
    "shift_aet"
    "awesome_align"
    "cl_lsa"
    "tssa"
)

SUCCESS_LIST=()
SKIPPED_LIST=()
FAILED_LIST=()

echo "========================================================================"
echo "    🚀 KHỞI ĐỘNG FULL BENCHMARK ViT5 (VietAI/vit5-base) TRÊN 3 NGÔN NGỮ"
echo "========================================================================"
echo "[*] Backbone Model : ${BACKBONE}"
echo "[*] Các phương pháp: ${MODELS[*]}"
echo "[*] Các ngôn ngữ   : ${LANGUAGES[*]}"
echo "[*] Tổng số mô hình: $((${#MODELS[@]} * ${#LANGUAGES[@]})) models (5 epochs, batch_size=${BATCH_SIZE}, lr=${LR})"
echo "========================================================================"

# Trap SIGINT to allow graceful stop of loop
trap 'echo -e "\n[!] Đã nhận tín hiệu dừng (Ctrl+C). Đang thoát..."; exit 1;' INT

for LANG in "${LANGUAGES[@]}"; do
    echo ""
    echo "===================================================================="
    echo ">>> BẮT ĐẦU BENCHMARK ViT5 CHO NGÔN NGỮ: ${LANG^^}"
    echo "===================================================================="
    
    for MODEL in "${MODELS[@]}"; do
        EXP_NAME="vit5_${MODEL}_${LANG}"
        CKPT_DIR="${OUTPUT_DIR}/${EXP_NAME}"
        PRED_FILE="${CKPT_DIR}/test_predictions.csv"
        
        # 1. Smart Skip nếu mô hình đã huấn luyện và đã đánh giá hoàn tất
        if [ -f "$PRED_FILE" ] && [ -s "$PRED_FILE" ]; then
            echo ""
            echo ">>> [⏭️ SMART SKIP] [${LANG^^}] ${EXP_NAME} ĐÃ HOÀN TẤT trước đó. Bỏ qua!"
            SKIPPED_LIST+=("${EXP_NAME}")
            continue
        fi

        echo ""
        echo ">>> [${LANG^^}] Đang huấn luyện: ${EXP_NAME} (Backbone: ${BACKBONE}) ..."
        START_TIME=$(date +%s)
        
        # Thiết lập cờ riêng cho TSSA vs Baselines
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
echo "                    📊 TỔNG KẾT TIẾN TRÌNH ViT5 BENCHMARK"
echo "========================================================================"
echo "  [+] Đã hoàn tất từ trước : ${#SKIPPED_LIST[@]} mô hình"
echo "  [+] Huấn luyện mới xong  : ${#SUCCESS_LIST[@]} mô hình"
echo "  [!] Bị lỗi               : ${#FAILED_LIST[@]} mô hình"
echo "========================================================================"

if [ ${#FAILED_LIST[@]} -eq 0 ]; then
    echo "🎉 CHÚC MỪNG: Toàn bộ 18 mô hình ViT5 đã hoàn tất thành công!"
    echo "💡 Bước tiếp theo: Chạy tổng kết và kiểm định thống kê:"
    echo "   python summary_vit5_results.py"
    echo "   python eval_significance_vit5.py"
else
    echo "⚠️ Chú ý: Một số mô hình gặp lỗi: ${FAILED_LIST[*]}"
fi
echo "========================================================================"
