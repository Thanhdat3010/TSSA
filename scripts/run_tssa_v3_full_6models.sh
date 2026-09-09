#!/usr/bin/env bash
# ==============================================================================
# TSSA 3.0 UNIVERSAL BENCHMARK SUITE: RUN ALL 6 MODELS (FULL ISOLATION)
# ==============================================================================
# Architectural Upgrades:
#   - Pareto Attention Capacity Budget: rho = 0.333 (1/3 Heads Specialized)
#   - Sharpened InfoNCE Temperature    : tau = 0.05
#   - Typology-Aware Lambda Scaling   :
#       * Isomorphic Group (Rhade, Tay) : struct=0.30, prime=0.15, route=0.05 (Sum=0.50)
#       * High-Fertility Group (Bahnar) : struct=0.30, prime=0.25, route=0.05 (Sum=0.60)
#
# Non-Destructive Preservation:
#   - 100% of legacy v1 and v2.1 checkpoints are preserved in checkpoints/
#   - TSSA 3.0 outputs are cleanly isolated in checkpoints/tssa_v3/
# ==============================================================================

set -e

# Global Configurations
NUM_EPOCHS=5
BATCH_SIZE=16
MAX_LEN=256
SEED=42
OUTPUT_DIR="checkpoints/tssa_v3"
TARGET_BUDGET=0.333
PRIME_TAU=0.05

LR_BARTPHO=2e-5
LR_VIT5=1e-4

LANGUAGES=("rhade" "tay" "bahnaric")

mkdir -p "${OUTPUT_DIR}"

echo "========================================================================"
echo "    🚀 TSSA 3.0 UNIVERSAL BENCHMARK: HUẤN LUYỆN 6/6 MÔ HÌNH CHUẨN MỰC"
echo "========================================================================"
echo "[*] Ngân sách Head (Pareto)   : rho = ${TARGET_BUDGET} (1/3 số đầu chú ý)"
echo "[*] Nhiệt độ InfoNCE (tau)     : tau = ${PRIME_TAU}"
echo "[*] Thư mục lưu trữ độc lập    : ${OUTPUT_DIR}"
echo "[*] Số lượng mô hình           : 6 (3 BARTpho + 3 ViT5)"
echo "[*] Ngôn ngữ                   : ${LANGUAGES[*]}"
echo "========================================================================"

trap 'echo -e "\n[!] Đã nhận tín hiệu dừng (Ctrl+C). Đang thoát an toàn..."; exit 1;' INT

# ------------------------------------------------------------------------------
# HÀM TRỢ GIÚP: CẤU HÌNH TYPOLOGY THEO NGÔN NGỮ
# ------------------------------------------------------------------------------
get_typology_lambdas() {
    local LANG=$1
    if [ "${LANG}" == "bahnaric" ]; then
        # Ngữ hệ Môn-Khơ Me (Phân mảnh cao kappa ~ 3.5): Tăng cường lực neo câu L_prime
        echo "0.30 0.25 0.05"
    else
        # Ngữ hệ Nam Đảo / Thái-Ka Đai (Ít phân mảnh kappa <= 1.8): Cân bằng từ vựng & câu
        echo "0.30 0.15 0.05"
    fi
}

# ------------------------------------------------------------------------------
# BƯỚC 1: HUẤN LUYỆN 3 MÔ HÌNH BARTpho (vinai/bartpho-syllable)
# ------------------------------------------------------------------------------
echo ""
echo "========================================================================"
echo ">>> [BƯỚC 1/2] BẮT ĐẦU HUẤN LUYỆN 3 MÔ HÌNH BARTpho (vinai/bartpho-syllable)"
echo "========================================================================"

for LANG in "${LANGUAGES[@]}"; do
    EXP_NAME="tssa_${LANG}"
    CKPT_DIR="${OUTPUT_DIR}/${EXP_NAME}"
    
    read -r L_STRUCT L_PRIME L_ROUTE <<< "$(get_typology_lambdas "${LANG}")"

    echo ""
    echo ">>> [BARTpho - ${LANG^^}] Khởi động: ${EXP_NAME} (LR=${LR_BARTPHO})"
    echo "    [*] Typology Lambdas: struct=${L_STRUCT}, prime=${L_PRIME}, route=${L_ROUTE} (tau=${PRIME_TAU}, rho=${TARGET_BUDGET})"
    
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
        --lambda_route "${L_ROUTE}" \
        --target_budget "${TARGET_BUDGET}" \
        --prime_tau "${PRIME_TAU}"

    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    echo ">>> [✅ THÀNH CÔNG] [BARTpho - ${LANG^^}] Hoàn tất ${EXP_NAME} trong ${DURATION}s!"
done

# ------------------------------------------------------------------------------
# BƯỚC 2: HUẤN LUYỆN 3 MÔ HÌNH ViT5 (VietAI/vit5-base)
# ------------------------------------------------------------------------------
echo ""
echo "========================================================================"
echo ">>> [BƯỚC 2/2] BẮT ĐẦU HUẤN LUYỆN 3 MÔ HÌNH ViT5 (VietAI/vit5-base)"
echo "========================================================================"

for LANG in "${LANGUAGES[@]}"; do
    EXP_NAME="vit5_tssa_${LANG}"
    CKPT_DIR="${OUTPUT_DIR}/${EXP_NAME}"

    read -r L_STRUCT L_PRIME L_ROUTE <<< "$(get_typology_lambdas "${LANG}")"

    echo ""
    echo ">>> [ViT5 - ${LANG^^}] Khởi động: ${EXP_NAME} (LR=${LR_VIT5})"
    echo "    [*] Typology Lambdas: struct=${L_STRUCT}, prime=${L_PRIME}, route=${L_ROUTE} (tau=${PRIME_TAU}, rho=${TARGET_BUDGET})"

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
        --lambda_route "${L_ROUTE}" \
        --target_budget "${TARGET_BUDGET}" \
        --prime_tau "${PRIME_TAU}"

    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    echo ">>> [✅ THÀNH CÔNG] [ViT5 - ${LANG^^}] Hoàn tất ${EXP_NAME} trong ${DURATION}s!"
done

# ------------------------------------------------------------------------------
# BƯỚC 3: BÁO CÁO NGHIỆM THU ĐỐI SOÁT 4 CHIỀU TOÀN DIỆN
# ------------------------------------------------------------------------------
echo ""
echo "========================================================================"
echo ">>> ĐANG TIẾN HÀNH TỔNG HỢP & IN BÁO CÁO ĐỐI SOÁT 4 CHIỀU (TSSA 3.0)..."
echo "========================================================================"
python compare_tssa_v3_full.py || true

echo ""
echo "========================================================================"
echo "    🎉 HOÀN TẤT TOÀN BỘ 6 MÔ HÌNH TSSA 3.0 UNIVERSAL BENCHMARK!"
echo "========================================================================"
