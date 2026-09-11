#!/usr/bin/env bash
# ==============================================================================
# UniTSSA 5.0 UNIVERSAL BENCHMARK SUITE (NAACL GOLD STANDARD): 6-MODEL RUNNER
# ==============================================================================
# Mathematical Formulation:
#   - Closed Capacity Budgeting     : rho*(H) = min(0.333, max(0.20, (H - 9)/H))
#       * BARTpho (H=16) -> rho* = 0.333 (11 Free Generation Heads)
#       * ViT5    (H=12) -> rho* = 0.250 (9 Free Generation Heads)
#   - Non-linear Typological Scaling: lambda_struct(kappa) = 0.36 * kappa^(-0.75)
#                                     lambda_prime(kappa)  = 0.12 * kappa^(-0.75)
#       * Isomorphic (Tay, Rhade : kappa <= 1.4)  -> struct=0.30, prime=0.10, route=0.05
#       * High-Fertility (Bahnar  : kappa ~ 3.5)   -> struct=0.14, prime=0.05, route=0.05
#   - Smoothed InfoNCE Temperature   : tau = 0.07 (Triệt tiêu hiện tượng sốc gradient)
#
# Non-Destructive Preservation:
#   - Toàn bộ checkpoints v1, v2.1, v3.0, v4.0 được bảo toàn nguyên vẹn 100%.
#   - Checkpoints UniTSSA 5.0 được lưu biệt lập tại checkpoints/tssa_v5/
# ==============================================================================

set -e

NUM_EPOCHS=5
BATCH_SIZE=16
MAX_LEN=256
SEED=42
OUTPUT_DIR="checkpoints/tssa_v5"
REPORT_FILE="${OUTPUT_DIR}/COMPARISON_REPORT.txt"
PRIME_TAU=0.07

LR_BARTPHO=2e-5
LR_VIT5=1e-4

LANGUAGES=("rhade" "tay" "bahnaric")

mkdir -p "${OUTPUT_DIR}"

echo "========================================================================"
echo "    🚀 UniTSSA 5.0 UNIVERSAL BENCHMARK (NAACL GOLD STANDARD): 6/6 MÔ HÌNH"
echo "========================================================================"
echo "[*] Nhiệt độ InfoNCE toàn cục    : tau = ${PRIME_TAU} (Smooth & Stable)"
echo "[*] Ngân sách BARTpho (H=16)     : rho* = 0.333 (11 Free Generation Heads)"
echo "[*] Ngân sách ViT5    (H=12)     : rho* = 0.250 (9 Free Generation Heads)"
echo "[*] Thư mục lưu trữ độc lập      : ${OUTPUT_DIR}"
echo "[*] Báo cáo nghiệm thu tự động   : ${REPORT_FILE}"
echo "[*] Danh sách ngôn ngữ           : ${LANGUAGES[*]}"
echo "========================================================================"

trap 'echo -e "\n[!] Đã nhận tín hiệu hủy (Ctrl+C). Đang dừng an toàn..."; exit 1;' INT

# ------------------------------------------------------------------------------
# HÀM TRỢ GIÚP: SUY LUẬN LAMBDAS THEO HÀM PHI TUYẾN KAPPA^(-0.75)
# ------------------------------------------------------------------------------
get_typological_lambdas() {
    local LANG=$1
    if [ "${LANG}" == "bahnaric" ]; then
        # Ngữ hệ Môn-Khơ Me (Phân mảnh cao kappa ~ 3.5):
        # 0.36 * 3.5^(-0.75) ~ 0.14; 0.12 * 3.5^(-0.75) ~ 0.05
        echo "0.14 0.05 0.05"
    else
        # Ngữ hệ Nam Đảo / Thái-Ka Đai (Đẳng cấu kappa <= 1.4):
        # 0.36 * 1.3^(-0.75) ~ 0.30; 0.12 * 1.3^(-0.75) ~ 0.10
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

    read -r L_STRUCT L_PRIME L_ROUTE <<< "$(get_typological_lambdas "${LANG}")"

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
    ELAPSED=$((END_TIME - START_TIME))
    echo "[✓] [BARTpho - ${LANG^^}] Hoàn tất sau ${ELAPSED}s. Đang cập nhật bảng đối soát..."

    # Cập nhật đối soát lũy tiến
    python compare_tssa_v5_full.py
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

    read -r L_STRUCT L_PRIME L_ROUTE <<< "$(get_typological_lambdas "${LANG}")"

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
    ELAPSED=$((END_TIME - START_TIME))
    echo "[✓] [ViT5 - ${LANG^^}] Hoàn tất sau ${ELAPSED}s. Đang cập nhật bảng đối soát..."

    # Cập nhật đối soát lũy tiến
    python compare_tssa_v5_full.py
done

# ------------------------------------------------------------------------------
# BƯỚC 3: NGHIỆM THU TOÀN DIỆN VÀ LƯU BÁO CÁO CUỐI CÙNG
# ------------------------------------------------------------------------------
echo ""
echo "========================================================================"
echo "    🎉 HOÀN TẤT TOÀN BỘ 6 MÔ HÌNH UniTSSA 5.0 BENCHMARK!"
echo "========================================================================"
python compare_tssa_v5_full.py | tee "${REPORT_FILE}"
echo ""
echo "[*] Báo cáo chi tiết đã được lưu trữ vĩnh viễn tại: ${REPORT_FILE}"
echo "========================================================================"
