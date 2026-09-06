#!/usr/bin/env bash
# ==============================================================================
# TSSA 2.1 CORE VERIFICATION GATE: 6-MODEL PILOT RUNNER
# ==============================================================================
# Scope:
#   - 3 Languages: Rhade (Ê Đê), Tay (Tày), Bahnaric (Ba Na)
#   - 2 Backbones: BARTpho-syllable (2e-5) & ViT5-base (1e-4)
#   - Optimal Weights (C_opt): lambda_struct=0.20, lambda_prime=0.10, lambda_route=0.05
#   - Preservation: 100% of vanilla and baseline checkpoints are untouched.
#   - Safety: Legacy TSSA v1 checkpoints are selectively archived to checkpoints/backup_tssa_legacy_v1/
# ==============================================================================

set -e

# Configuration
NUM_EPOCHS=5
BATCH_SIZE=16
MAX_LEN=256
SEED=42
OUTPUT_DIR="checkpoints"
BACKUP_DIR="checkpoints/backup_tssa_legacy_v1"

# Hyperparameters
LR_BARTPHO=2e-5
LR_VIT5=1e-4
L_STRUCT=0.20
L_PRIME=0.10
L_ROUTE=0.05

LANGUAGES=("rhade" "tay" "bahnaric")

echo "========================================================================"
echo "    🚀 TSSA 2.1 VERIFICATION GATE: CHẠY THỬ NGHIỆM 6 MÔ HÌNH CỐT LÕI"
echo "========================================================================"
echo "[*] Bộ trọng số tối ưu (C_opt): struct=${L_STRUCT}, prime=${L_PRIME}, route=${L_ROUTE} (Sum=0.35)"
echo "[*] Số ngôn ngữ              : ${LANGUAGES[*]}"
echo "[*] Tổng số mô hình           : 6 (3 BARTpho + 3 ViT5)"
echo "[*] Thư mục lưu trữ          : ${OUTPUT_DIR}"
echo "[*] Thư mục sao lưu TSSA v1  : ${BACKUP_DIR}"
echo "========================================================================"

trap 'echo -e "\n[!] Đã nhận tín hiệu hủy (Ctrl+C). Đang dừng an toàn..."; exit 1;' INT

# ------------------------------------------------------------------------------
# BƯỚC 0: SAO LƯU CHỌN LỌC CÁC CHECKPOINT TSSA CŨ (GIỮ NGUYÊN TOÀN BỘ BASELINES)
# ------------------------------------------------------------------------------
echo ""
echo ">>> [BƯỚC 0/3] ĐANG TIẾN HÀNH SAO LƯU CHỌN LỌC & ĐỐI SOÁT CHECKPOINTS TSSA v1 CŨ..."
bash scripts/backup_legacy_tssa.sh
echo "[✓] BƯỚC 0 HOÀN TẤT: Toàn bộ baseline và vanilla checkpoints được giữ nguyên 100%!"

# ------------------------------------------------------------------------------
# BƯỚC 1: HUẤN LUYỆN 3 MÔ HÌNH BARTpho VỚI TSSA 2.1 (C_opt)
# ------------------------------------------------------------------------------
echo ""
echo "========================================================================"
echo ">>> [BƯỚC 1/3] BẮT ĐẦU HUẤN LUYỆN 3 MÔ HÌNH BARTpho (vinai/bartpho-syllable)"
echo "========================================================================"

for LANG in "${LANGUAGES[@]}"; do
    EXP_NAME="tssa_${LANG}"
    CKPT_DIR="${OUTPUT_DIR}/${EXP_NAME}"
    echo ""
    echo ">>> [BARTpho - ${LANG^^}] Khởi động huấn luyện: ${EXP_NAME} (LR=${LR_BARTPHO}) ..."
    START_TIME=$(date +%s)
    
    python train.py \
        --lang "${LANG}" \
        --model_ckpt "vinai/bartpho-syllable" \
        --model_type "tssa" \
        --exp_name "${EXP_NAME}" \
        --output_dir "${OUTPUT_DIR}" \
        --num_epochs "${NUM_EPOCHS}" \
        --batch_size "${BATCH_SIZE}" \
        --learning_rate "${LR_BARTPHO}" \
        --max_source_length "${MAX_LEN}" \
        --max_target_length "${MAX_LEN}" \
        --seed "${SEED}" \
        --fp16 \
        --use_struct \
        --use_prime \
        --use_route \
        --lambda_struct "${L_STRUCT}" \
        --lambda_prime "${L_PRIME}" \
        --lambda_route "${L_ROUTE}"

    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    echo ">>> [✅ THÀNH CÔNG] [BARTpho - ${LANG^^}] Hoàn tất ${EXP_NAME} trong ${DURATION}s!"
done

# ------------------------------------------------------------------------------
# BƯỚC 2: HUẤN LUYỆN 3 MÔ HÌNH ViT5 VỚI TSSA 2.1 (C_opt)
# ------------------------------------------------------------------------------
echo ""
echo "========================================================================"
echo ">>> [BƯỚC 2/3] BẮT ĐẦU HUẤN LUYỆN 3 MÔ HÌNH ViT5 (VietAI/vit5-base)"
echo "========================================================================"

for LANG in "${LANGUAGES[@]}"; do
    EXP_NAME="vit5_tssa_${LANG}"
    CKPT_DIR="${OUTPUT_DIR}/${EXP_NAME}"
    echo ""
    echo ">>> [ViT5 - ${LANG^^}] Khởi động huấn luyện: ${EXP_NAME} (LR=${LR_VIT5}) ..."
    START_TIME=$(date +%s)

    python train.py \
        --lang "${LANG}" \
        --model_ckpt "VietAI/vit5-base" \
        --model_type "tssa" \
        --exp_name "${EXP_NAME}" \
        --output_dir "${OUTPUT_DIR}" \
        --num_epochs "${NUM_EPOCHS}" \
        --batch_size "${BATCH_SIZE}" \
        --learning_rate "${LR_VIT5}" \
        --max_source_length "${MAX_LEN}" \
        --max_target_length "${MAX_LEN}" \
        --seed "${SEED}" \
        --fp16 \
        --use_struct \
        --use_prime \
        --use_route \
        --lambda_struct "${L_STRUCT}" \
        --lambda_prime "${L_PRIME}" \
        --lambda_route "${L_ROUTE}"

    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    echo ">>> [✅ THÀNH CÔNG] [ViT5 - ${LANG^^}] Hoàn tất ${EXP_NAME} trong ${DURATION}s!"
done

# ------------------------------------------------------------------------------
# BƯỚC 3: ĐÁNH GIÁ TỔNG KẾT & KIỂM ĐỊNH Ý NGHĨA THỐNG KÊ
# ------------------------------------------------------------------------------
echo ""
echo "========================================================================"
echo ">>> [BƯỚC 3/3] ĐANG TIẾN HÀNH TỔNG HỢP KẾT QUẢ & KIỂM ĐỊNH THỐNG KÊ..."
echo "========================================================================"

echo ""
echo "--- [1/4] Tổng hợp kết quả BARTpho ---"
python summary_results.py --comet || true

echo ""
echo "--- [2/4] Kiểm định thống kê Paired Bootstrap BARTpho ---"
python eval_significance.py || true

echo ""
echo "--- [3/4] Tổng hợp kết quả ViT5 ---"
python summary_vit5_results.py --comet || true

echo ""
echo "--- [4/4] Kiểm định thống kê Paired Bootstrap ViT5 ---"
python eval_significance_vit5.py || true

echo ""
echo "--- [ĐẶC BIỆT] Bảng so sánh trực diện TSSA 2.1 vs TSSA v1 Cũ vs Vanilla ---"
python compare_tssa_v2_pilot.py || true

echo ""
echo "========================================================================"
echo "    🛑 VERIFICATION GATE: HOÀN TẤT HUẤN LUYỆN & ĐÁNH GIÁ 6 MÔ HÌNH"
echo "========================================================================"
echo "  [!] QUY TẮC BẮT BUỘC:"
echo "      - Dừng lại tại đây để kiểm tra số liệu so sánh hiển thị phía trên."
echo "      - KHÔNG tự động rollback hay ghi đè checkpoints."
echo "      - KHÔNG tự ý chạy ablation study."
echo "      - Gửi kết quả bảng so sánh cho trợ lý AI để cùng đánh giá và quyết định bước tiếp theo."
echo "========================================================================"
