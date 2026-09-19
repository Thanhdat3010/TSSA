#!/usr/bin/env bash
# ==============================================================================
# TSSA-PRO UNIVERSAL BENCHMARK (6-MODEL UNIFIED RUNNER)
# ==============================================================================
# Fair Comparison Protocol:
#   - Identical standard LR: LR_BARTpho=2e-5, LR_ViT5=1e-4
#   - Epochs: 5, Batch Size: 16, Seed: 42, FP16
#   - Scale-Invariant Latent Hypersphere (S^{D-1}) with Batch-Centering (Anti-Anisotropy)
#   - Length-Normalized Dynamic Information Entropy Filtering (tau_H = 0.50)
#   - Continuous Gaussian Fertility Protection (Ba Na gracefully protected)
#   - Decoder: 100% Free Autoregressive Syntax Generation
#
# Checkpoints saved strictly to: checkpoints/tssa_pro/
# ==============================================================================

set -e

NUM_EPOCHS=5
BATCH_SIZE=16
MAX_LEN=256
SEED=42
OUTPUT_DIR="checkpoints/tssa_pro"
REPORT_FILE="${OUTPUT_DIR}/COMPARISON_REPORT.txt"

LR_BARTPHO=2e-5
LR_VIT5=1e-4

PRIME_TAU=0.07
CONF_THRESHOLD=0.20
ENTROPY_TAU=0.50

# Allow LAMBDA_STRUCT to be overridden from argument $1 or env var, default 1.00
LAMBDA_STRUCT="${1:-${LAMBDA_STRUCT:-1.00}}"
LAMBDA_PRIME=0.08
LAMBDA_ROUTE=0.00
SIGMA_KAPPA=0.75

LANGUAGES=("rhade" "tay" "bahnaric")

mkdir -p "${OUTPUT_DIR}"

echo "========================================================================"
echo "    🚀 TSSA-PRO UNIVERSAL BENCHMARK RUNNER (6/6 MÔ HÌNH TUẦN TỰ)"
echo "========================================================================"
echo "[*] Thư mục lưu trữ mới       : ${OUTPUT_DIR}"
echo "[*] Trọng số mỏ neo           : struct=${LAMBDA_STRUCT}, prime=${LAMBDA_PRIME}"
echo "[*] Tốc độ học (Fair Standard): BARTpho=${LR_BARTPHO}, ViT5=${LR_VIT5}"
echo "[*] Số epochs                 : ${NUM_EPOCHS}, Batch: ${BATCH_SIZE}, Seed: ${SEED}"
echo "[*] Cơ chế toán học           : Centering + Scale-Invariant + tau_H=${ENTROPY_TAU}"
echo "[*] Danh sách ngôn ngữ        : ${LANGUAGES[*]}"
echo "========================================================================"

trap 'echo -e "\n[!] Đã nhận tín hiệu hủy (Ctrl+C). Đang dừng an toàn..."; exit 1;' INT

# ------------------------------------------------------------------------------
# PHẦN 1: HUẤN LUYỆN 3 MÔ HÌNH BARTpho (vinai/bartpho-syllable)
# ------------------------------------------------------------------------------
echo ""
echo "========================================================================"
echo ">>> [BƯỚC 1/2] BẮT ĐẦU HUẤN LUYỆN 3 MÔ HÌNH BARTpho (SCALE-INVARIANT)"
echo "========================================================================"

for LANG in "${LANGUAGES[@]}"; do
    EXP_NAME="tssa_${LANG}"
    CKPT_DIR="${OUTPUT_DIR}/${EXP_NAME}"

    echo ""
    echo ">>> [BARTpho - ${LANG^^}] Bắt đầu: ${EXP_NAME} (LR=${LR_BARTPHO})"
    echo "    [*] Cấu hình: struct=${LAMBDA_STRUCT}, prime=${LAMBDA_PRIME}, tau_H=${ENTROPY_TAU}"

    START_TIME=$(date +%s)

    python train_tssa_pro.py \
        --lang "${LANG}" \
        --model_ckpt "vinai/bartpho-syllable" \
        --exp_name "${EXP_NAME}" \
        --output_dir "${OUTPUT_DIR}" \
        --num_epochs "${NUM_EPOCHS}" \
        --batch_size "${BATCH_SIZE}" \
        --learning_rate "${LR_BARTPHO}" \
        --max_source_length "${MAX_LEN}" \
        --max_target_length "${MAX_LEN}" \
        --seed "${SEED}" \
        --fp16 \
        --sigma_kappa "${SIGMA_KAPPA}" \
        --use_struct \
        --use_prime \
        --no_route \
        --use_centering \
        --protect_struct_fertility \
        --entropy_tau "${ENTROPY_TAU}" \
        --lambda_struct "${LAMBDA_STRUCT}" \
        --lambda_prime "${LAMBDA_PRIME}" \
        --lambda_route "${LAMBDA_ROUTE}" \
        --prime_tau "${PRIME_TAU}" \
        --conf_threshold "${CONF_THRESHOLD}"

    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    echo "[✓] [BARTpho - ${LANG^^}] Hoàn tất sau ${ELAPSED} giây."
done

# ------------------------------------------------------------------------------
# PHẦN 2: HUẤN LUYỆN 3 MÔ HÌNH ViT5 (VietAI/vit5-base)
# ------------------------------------------------------------------------------
echo ""
echo "========================================================================"
echo ">>> [BƯỚC 2/2] BẮT ĐẦU HUẤN LUYỆN 3 MÔ HÌNH ViT5 (SCALE-INVARIANT)"
echo "========================================================================"

for LANG in "${LANGUAGES[@]}"; do
    EXP_NAME="vit5_tssa_${LANG}"
    CKPT_DIR="${OUTPUT_DIR}/${EXP_NAME}"

    echo ""
    echo ">>> [ViT5 - ${LANG^^}] Bắt đầu: ${EXP_NAME} (LR=${LR_VIT5})"
    echo "    [*] Cấu hình: struct=${LAMBDA_STRUCT}, prime=${LAMBDA_PRIME}, tau_H=${ENTROPY_TAU}"

    START_TIME=$(date +%s)

    python train_tssa_pro.py \
        --lang "${LANG}" \
        --model_ckpt "VietAI/vit5-base" \
        --exp_name "${EXP_NAME}" \
        --output_dir "${OUTPUT_DIR}" \
        --num_epochs "${NUM_EPOCHS}" \
        --batch_size "${BATCH_SIZE}" \
        --learning_rate "${LR_VIT5}" \
        --max_source_length "${MAX_LEN}" \
        --max_target_length "${MAX_LEN}" \
        --seed "${SEED}" \
        --fp16 \
        --sigma_kappa "${SIGMA_KAPPA}" \
        --use_struct \
        --use_prime \
        --no_route \
        --use_centering \
        --protect_struct_fertility \
        --entropy_tau "${ENTROPY_TAU}" \
        --lambda_struct "${LAMBDA_STRUCT}" \
        --lambda_prime "${LAMBDA_PRIME}" \
        --lambda_route "${LAMBDA_ROUTE}" \
        --prime_tau "${PRIME_TAU}" \
        --conf_threshold "${CONF_THRESHOLD}"

    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    echo "[✓] [ViT5 - ${LANG^^}] Hoàn tất sau ${ELAPSED} giây."
done

# ------------------------------------------------------------------------------
# PHẦN 3: TỰ ĐỘNG ĐỐI SOÁT & XUẤT BÁO CÁO NGHIỆM THU
# ------------------------------------------------------------------------------
echo ""
echo "========================================================================"
echo ">>> ĐANG TIẾN HÀNH ĐỐI SOÁT KẾT QUẢ VỚI TOÀN BỘ BASELINES..."
echo "========================================================================"

python compare_tssa_pro_vs_baselines.py

echo ""
echo "🎉 [HOÀN TẤT TOÀN DIỆN] 6/6 MÔ HÌNH TSSA-PRO ĐÃ ĐƯỢC HUẤN LUYỆN VÀ ĐỐI SOÁT!"
echo "Báo cáo chi tiết xem tại: ${REPORT_FILE}"
