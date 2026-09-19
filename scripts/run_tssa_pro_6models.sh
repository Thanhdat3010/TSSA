#!/usr/bin/env bash
# ==============================================================================
# TSSA-PRO UNIVERSAL BENCHMARK (6-MODEL UNIFIED RUNNER)
# ==============================================================================
# Fair Comparison Protocol:
#   - Identical standard LR: LR_BARTpho=2e-5, LR_ViT5=1e-4
#   - Epochs: 5, Batch Size: 16, Seed: 42, FP16
#   - Closed Capacity Budget: rho* = 0.250 (25% Anchor Heads, 75% Free Heads)
#   - Continuous Gaussian Fertility Attenuation (via Typological Kappa)
#   - Dynamic Information-Theoretic Entropy Filtering (Latent Barycenter)
#
# Checkpoints saved strictly to: checkpoints/tssa_pro/
# All legacy & previous checkpoints remain 100% preserved.
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
TARGET_BUDGET=0.250

LANGUAGES=("rhade" "tay" "bahnaric")

mkdir -p "${OUTPUT_DIR}"

echo "========================================================================"
echo "    🚀 TSSA-PRO UNIVERSAL BENCHMARK RUNNER (6/6 MÔ HÌNH TUẦN TỰ)"
echo "========================================================================"
echo "[*] Thư mục lưu trữ mới       : ${OUTPUT_DIR}"
echo "[*] Ngân sách đầu chú ý rho*  : ${TARGET_BUDGET} (25% Anchor Heads, 75% Free)"
echo "[*] Tốc độ học (Fair Standard): BARTpho=${LR_BARTPHO}, ViT5=${LR_VIT5}"
echo "[*] Số epochs                 : ${NUM_EPOCHS}, Batch: ${BATCH_SIZE}, Seed: ${SEED}"
echo "[*] Danh sách ngôn ngữ        : ${LANGUAGES[*]}"
echo "========================================================================"

trap 'echo -e "\n[!] Đã nhận tín hiệu hủy (Ctrl+C). Đang dừng an toàn..."; exit 1;' INT

# ------------------------------------------------------------------------------
# THAM SỐ TOÁN HỌC PHỔ QUÁT TSSA (SCALE-INVARIANT HYPERSPHERE - ZERO IF/ELSE)
# ------------------------------------------------------------------------------
# Toàn bộ tham số được cố định theo lý thuyết hình học mặt cầu bất biến thang đo:
#   - L_struct: lambda = 0.20, Entropy Gate tau_H = 1.5, Cosine Hypersphere S^{D-1}
#   - L_prime : lambda = 0.08, tau_prime = 0.07, sigma_kappa = 0.75
#               Hệ số Gaussian suy giảm InfoNCE: exp(-(max(1, kappa)-1)^2 / (2*0.75^2))
#   - Decoder : 100% Tự do sinh câu cú pháp (Zero intervention on Decoder)
#   - Precision: FP16 tiêu chuẩn (Loss & Softmax tính bằng FP32 chống underflow)
# ------------------------------------------------------------------------------
LAMBDA_STRUCT=0.20
LAMBDA_PRIME=0.08
LAMBDA_ROUTE=0.00
SIGMA_KAPPA=0.75

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
    echo "    [*] Cấu hình toán học phổ quát: struct=${LAMBDA_STRUCT}, prime=${LAMBDA_PRIME}, route=${LAMBDA_ROUTE}"

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
    echo "    [*] Cấu hình toán học phổ quát: struct=${LAMBDA_STRUCT}, prime=${LAMBDA_PRIME}, route=${LAMBDA_ROUTE}"

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
