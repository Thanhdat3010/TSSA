#!/usr/bin/env bash
# ==============================================================================
# UniTSSA FINAL UNIVERSAL BENCHMARK (NAACL GOLD STANDARD): 6-MODEL RUNNER
# ==============================================================================
# Mathematical Formulation:
#   - Closed Capacity Budgeting          : rho*(H) = min(0.333, max(0.20, (H - 9)/H))
#       * BARTpho (H=16) -> rho* = 0.333 (11 Free Generation Heads)
#       * ViT5    (H=12) -> rho* = 0.250 (9 Free Generation Heads)
#   - Confidence-Sharpened Structural Anchoring (Anchor Novelty 100% Preserved):
#       * Alignment Softmax Sharpening   : tau_align = 0.10 (EACL 2021 AWESOME-align Standard)
#       * Selective Anchor Gating        : conf_threshold = 0.25 (EMNLP 2019 Standard)
#       * Continuous Anchor Weighting    : lambda_struct(kappa) = 0.12 + 0.20 * exp(-(kappa-1)^2 / 2.0)
#   - Continuous Hyperbolic Priming Compass:
#       * lambda_prime(kappa)            : 0.08 + 0.06 * tanh(kappa - 1)
#   - Controlled Fair Learning Rates     : LR_BARTpho=2e-5, LR_ViT5=1e-4 (100% Frozen & Fair)
#
# Non-Destructive Preservation:
#   - Toàn bộ checkpoints v1, v2.1, v3.0, v4.0, v5.0, v6.0 được bảo toàn nguyên vẹn 100%.
#   - Checkpoints UniTSSA Final được lưu biệt lập tại checkpoints/tssa_final/
# ==============================================================================

set -e

NUM_EPOCHS=5
BATCH_SIZE=16
MAX_LEN=256
SEED=42
OUTPUT_DIR="checkpoints/tssa_final"
REPORT_FILE="${OUTPUT_DIR}/COMPARISON_REPORT.txt"
PRIME_TAU=0.07
CONF_THRESHOLD=0.20

# Fair Comparison: 100% Identical to baselines
LR_BARTPHO=2e-5
LR_VIT5=1e-4

LANGUAGES=("rhade" "tay" "bahnaric")

mkdir -p "${OUTPUT_DIR}"

echo "========================================================================"
echo "    🚀 UniTSSA FINAL BENCHMARK (NAACL GOLD STANDARD): 6/6 MÔ HÌNH"
echo "========================================================================"
echo "[*] Nhiệt độ căn chỉnh Attention : tau_align = 0.10 (Làm sắc nhọn đỉnh mỏ neo)"
echo "[*] Ngưỡng lọc mỏ neo chọn lọc   : conf_threshold = ${CONF_THRESHOLD} (Gating mỏ neo sạch)"
echo "[*] Nhiệt độ InfoNCE toàn cục    : tau = ${PRIME_TAU} (Smooth & Stable)"
echo "[*] Ngân sách BARTpho (H=16)     : rho* = 0.333 (11 Free Generation Heads)"
echo "[*] Ngân sách ViT5    (H=12)     : rho* = 0.250 (9 Free Generation Heads)"
echo "[*] Tốc độ học (Fair Standard)   : BARTpho=${LR_BARTPHO}, ViT5=${LR_VIT5}"
echo "[*] Thư mục lưu trữ độc lập      : ${OUTPUT_DIR}"
echo "[*] Báo cáo nghiệm thu tự động   : ${REPORT_FILE}"
echo "[*] Danh sách ngôn ngữ           : ${LANGUAGES[*]}"
echo "========================================================================"

trap 'echo -e "\n[!] Đã nhận tín hiệu hủy (Ctrl+C). Đang dừng an toàn..."; exit 1;' INT

# ------------------------------------------------------------------------------
# HÀM TRỢ GIÚP: SUY LUẬN LAMBDAS THEO ĐỊNH LUẬT LIÊN TỤC TOÀN NĂNG
# ------------------------------------------------------------------------------
get_typological_lambdas_final() {
    local LANG=$1
    if [ "${LANG}" == "bahnaric" ]; then
        # Ba Na (Phân mảnh cao kappa ~ 3.5):
        # struct = 0.12 + 0.20 * exp(-3.125) = 0.13 (ANCHOR HOẠT ĐỘNG CHỌN LỌC!)
        # prime  = 0.08 + 0.06 * tanh(2.5)   = 0.14 (LA BÀN CÂU VỮNG CHẮC!)
        # route  = 0.05
        echo "0.13 0.14 0.05"
    elif [ "${LANG}" == "rhade" ]; then
        # Ê Đê (Đẳng cấu kappa ~ 1.4):
        # struct = 0.12 + 0.20 * exp(-0.08)  = 0.30 (Đỉnh cao cấu trúc)
        # prime  = 0.08 + 0.06 * tanh(0.4)   = 0.10
        # route  = 0.05
        echo "0.30 0.10 0.05"
    else
        # Tày (Đẳng cấu kappa ~ 1.2):
        # struct = 0.12 + 0.20 * exp(-0.02)  = 0.32 (Đỉnh cao cấu trúc)
        # prime  = 0.08 + 0.06 * tanh(0.2)   = 0.09
        # route  = 0.05
        echo "0.32 0.09 0.05"
    fi
}

# ------------------------------------------------------------------------------
# BƯỚC 1: HUẤN LUYỆN 3 MÔ HÌNH BARTpho (H=16, rho=0.333, LR=2e-5)
# ------------------------------------------------------------------------------
echo ""
echo "========================================================================"
echo ">>> [BƯỚC 1/2] BẮT ĐẦU HUẤN LUYỆN 3 MÔ HÌNH BARTpho (vinai/bartpho-syllable)"
echo "========================================================================"

RHO_BARTPHO=0.333

for LANG in "${LANGUAGES[@]}"; do
    EXP_NAME="tssa_${LANG}"
    CKPT_DIR="${OUTPUT_DIR}/${EXP_NAME}"

    read -r L_STRUCT L_PRIME L_ROUTE <<< "$(get_typological_lambdas_final "${LANG}")"

    echo ""
    echo ">>> [BARTpho - ${LANG^^}] Khởi động: ${EXP_NAME} (LR=${LR_BARTPHO})"
    echo "    [*] Calibration: rho=${RHO_BARTPHO}, struct=${L_STRUCT}, prime=${L_PRIME}, route=${L_ROUTE}, c_th=${CONF_THRESHOLD}, tau=${PRIME_TAU}"

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
        --prime_tau "${PRIME_TAU}" \
        --conf_threshold "${CONF_THRESHOLD}"

    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    echo "[✓] [BARTpho - ${LANG^^}] Hoàn tất sau ${ELAPSED}s. Đang cập nhật bảng đối soát..."

    # Cập nhật đối soát lũy tiến
    python compare_unitssa_final_full.py
done

# ------------------------------------------------------------------------------
# BƯỚC 2: HUẤN LUYỆN 3 MÔ HÌNH ViT5 (H=12, rho=0.250, LR=1e-4)
# ------------------------------------------------------------------------------
echo ""
echo "========================================================================"
echo ">>> [BƯỚC 2/2] BẮT ĐẦU HUẤN LUYỆN 3 MÔ HÌNH ViT5 (VietAI/vit5-base)"
echo "========================================================================"

RHO_VIT5=0.250

for LANG in "${LANGUAGES[@]}"; do
    EXP_NAME="vit5_tssa_${LANG}"
    CKPT_DIR="${OUTPUT_DIR}/${EXP_NAME}"

    read -r L_STRUCT L_PRIME L_ROUTE <<< "$(get_typological_lambdas_final "${LANG}")"

    echo ""
    echo ">>> [ViT5 - ${LANG^^}] Khởi động: ${EXP_NAME} (LR=${LR_VIT5})"
    echo "    [*] Calibration: rho=${RHO_VIT5}, struct=${L_STRUCT}, prime=${L_PRIME}, route=${L_ROUTE}, c_th=${CONF_THRESHOLD}, tau=${PRIME_TAU}"

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
        --prime_tau "${PRIME_TAU}" \
        --conf_threshold "${CONF_THRESHOLD}"

    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    echo "[✓] [ViT5 - ${LANG^^}] Hoàn tất sau ${ELAPSED}s. Đang cập nhật bảng đối soát..."

    # Cập nhật đối soát lũy tiến
    python compare_unitssa_final_full.py
done

# ------------------------------------------------------------------------------
# BƯỚC 3: NGHIỆM THU TOÀN DIỆN VÀ LƯU BÁO CÁO CUỐI CÙNG
# ------------------------------------------------------------------------------
echo ""
echo "========================================================================"
echo "    🎉 HOÀN TẤT TOÀN BỘ 6 MÔ HÌNH UniTSSA FINAL BENCHMARK!"
echo "========================================================================"
python compare_unitssa_final_full.py | tee "${REPORT_FILE}"
echo ""
echo "[*] Báo cáo chi tiết đã được lưu trữ vĩnh viễn tại: ${REPORT_FILE}"
echo "========================================================================"
