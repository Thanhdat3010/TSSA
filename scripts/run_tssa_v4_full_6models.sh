#!/usr/bin/env bash
# ==============================================================================
# UniTSSA 4.0 UNIVERSAL BENCHMARK SUITE (NAACL GRADE): 6-MODEL RUNNER
# ==============================================================================
# Mathematical Formulation:
#   - Closed Capacity Budgeting   : rho*(H) = min(0.333, max(0.20, (H - 9)/H))
#       * BARTpho (H=16) -> rho* = 0.333 (11 Free Generation Heads)
#       * ViT5    (H=12) -> rho* = 0.250 (9 Free Generation Heads)
#   - Fertility-Calibrated Scaling : lambda_struct(kappa) = 0.35 / (1 + 0.5 * ln(kappa))
#       * Isomorphic (Tay, Rhade : kappa <= 1.4)  -> struct=0.30, prime=0.10, route=0.05
#       * High-Fertility (Bahnar  : kappa ~ 3.5)   -> struct=0.20, prime=0.10, route=0.05
#   - Smoothed InfoNCE Temperature : tau = 0.07 (Triệt tiêu hiện tượng sốc gradient)
#
# Non-Destructive Preservation:
#   - Toàn bộ checkpoints v1, v2.1, v3.0 được bảo toàn nguyên vẹn 100%.
#   - Checkpoints UniTSSA 4.0 được lưu biệt lập tại checkpoints/tssa_v4/
# ==============================================================================

set -e

NUM_EPOCHS=5
BATCH_SIZE=16
MAX_LEN=256
SEED=42
OUTPUT_DIR="checkpoints/tssa_v4"
PRIME_TAU=0.07

LR_BARTPHO=2e-5
LR_VIT5=1e-4

LANGUAGES=("rhade" "tay" "bahnaric")

mkdir -p "${OUTPUT_DIR}"

echo "========================================================================"
echo "    🚀 UniTSSA 4.0 UNIVERSAL BENCHMARK (NAACL GRADE): CHẠY 6/6 MÔ HÌNH"
echo "========================================================================"
echo "[*] Nhiệt độ InfoNCE toàn cục  : tau = ${PRIME_TAU} (Ổn định mượt mà)"
echo "[*] Ngân sách BARTpho (H=16)   : rho* = 0.333 (11 Free Generation Heads)"
echo "[*] Ngân sách ViT5    (H=12)   : rho* = 0.250 (9 Free Generation Heads)"
echo "[*] Thư mục lưu trữ độc lập    : ${OUTPUT_DIR}"
echo "[*] Danh sách ngôn ngữ         : ${LANGUAGES[*]}"
echo "========================================================================"

trap 'echo -e "\n[!] Đã nhận tín hiệu hủy (Ctrl+C). Đang dừng an toàn..."; exit 1;' INT

# ------------------------------------------------------------------------------
# HÀM TRỢ GIÚP: SUY LUẬN LAMBDAS THEO ĐỘ PHÂN MẢNH TỪ TỐ (KAPPA)
# ------------------------------------------------------------------------------
get_fertility_lambdas() {
    local LANG=$1
    if [ "${LANG}" == "bahnaric" ]; then
        # Ngữ hệ Môn-Khơ Me (Phân mảnh cao kappa ~ 3.5): lambda_struct = 0.20
        echo "0.20 0.10 0.05"
    else
        # Ngữ hệ Nam Đảo / Thái-Ka Đai (Đẳng cấu kappa <= 1.4): lambda_struct = 0.30
        echo "0.30 0.10 0.05"
    fi
}

# ------------------------------------------------------------------------------
# BƯỚC 1: HUẤN LUYỆN 3 MÔ HÌNH BARTpho (H=16, rho=0.333)
# ------------------------------------------------------------------------------
echo ""
echo "========================================================================"
echo ">>> [BƯỚC 1/2] BẮT ĐẦU HUẤN LUYỆN 3 MÔ HÌNH BARTpho (vinai/bartpho-syllable)"
echo "========================================================================"

RHO_BARTPHO=0.333

for LANG in "${LANGUAGES[@]}"; do
    EXP_NAME="tssa_${LANG}"
    CKPT_DIR="${OUTPUT_DIR}/${EXP_NAME}"

    read -r L_STRUCT L_PRIME L_ROUTE <<< "$(get_fertility_lambdas "${LANG}")"

    echo ""
    echo ">>> [BARTpho - ${LANG^^}] Khởi động: ${EXP_NAME} (LR=${LR_BARTPHO})"
    echo "    [*] Calibration: rho=${RHO_BARTPHO}, struct=${L_STRUCT}, prime=${L_PRIME}, route=${L_ROUTE}, tau=${PRIME_TAU}"

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
        --target_budget "${RHO_BARTPHO}" \
        --prime_tau "${PRIME_TAU}"

    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    echo ">>> [✅ THÀNH CÔNG] [BARTpho - ${LANG^^}] Hoàn tất ${EXP_NAME} trong ${DURATION}s!"
done

# ------------------------------------------------------------------------------
# BƯỚC 2: HUẤN LUYỆN 3 MÔ HÌNH ViT5 (H=12, rho=0.250)
# ------------------------------------------------------------------------------
echo ""
echo "========================================================================"
echo ">>> [BƯỚC 2/2] BẮT ĐẦU HUẤN LUYỆN 3 MÔ HÌNH ViT5 (VietAI/vit5-base)"
echo "========================================================================"

RHO_VIT5=0.250

for LANG in "${LANGUAGES[@]}"; do
    EXP_NAME="vit5_tssa_${LANG}"
    CKPT_DIR="${OUTPUT_DIR}/${EXP_NAME}"

    read -r L_STRUCT L_PRIME L_ROUTE <<< "$(get_fertility_lambdas "${LANG}")"

    echo ""
    echo ">>> [ViT5 - ${LANG^^}] Khởi động: ${EXP_NAME} (LR=${LR_VIT5})"
    echo "    [*] Calibration: rho=${RHO_VIT5}, struct=${L_STRUCT}, prime=${L_PRIME}, route=${L_ROUTE}, tau=${PRIME_TAU}"

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
        --target_budget "${RHO_VIT5}" \
        --prime_tau "${PRIME_TAU}"

    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    echo ">>> [✅ THÀNH CÔNG] [ViT5 - ${LANG^^}] Hoàn tất ${EXP_NAME} trong ${DURATION}s!"
done

# ------------------------------------------------------------------------------
# BƯỚC 3: BÁO CÁO NGHIỆM THU ĐỐI SOÁT 5 CHIỀU TOÀN DIỆN
# ------------------------------------------------------------------------------
echo ""
echo "========================================================================"
echo ">>> ĐANG TIẾN HÀNH TỔNG HỢP & IN BÁO CÁO ĐỐI SOÁT 5 CHIỀU (UniTSSA 4.0)..."
echo "========================================================================"
python compare_tssa_v4_full.py || true

echo ""
echo "========================================================================"
echo "    🎉 HOÀN TẤT TOÀN BỘ 6 MÔ HÌNH UniTSSA 4.0 UNIVERSAL BENCHMARK!"
echo "========================================================================"
