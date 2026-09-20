#!/usr/bin/env bash
# ==============================================================================
# TSSA-PRO: 3-SEED RIGOROUS VALIDATION SUITE (SEEDS 42, 43, 44)
# ==============================================================================
# Pre-registered Go/No-go Protocol as requested by Senior Peer Review:
# 1. 3 Seeds: 42, 43, 44
# 2. 3 Core Settings:
#    - BARTpho Tày (tay -> vi)
#    - BARTpho Ê Đê (rhade -> vi)
#    - ViT5 Tày (tay -> vi)
# 3. Dedicated RNG Control for BARTpho Tày:
#    - lambda = 0.0 with teacher forward executed to rule out RNG seed artifacts
# 4. Strictly isolated output folder:
#    - checkpoints/tssa_seeds/
#
# Usage:
#   bash scripts/run_3seed_validation.sh [all | bartpho_tay | bartpho_rhade | vit5_tay]
# ==============================================================================

set -e

TARGET="${1:-all}"
SEEDS=(42 43 44)
OUTPUT_DIR="checkpoints/tssa_seeds"
REPORT_SCRIPT="scripts/report_3seed_validation.py"

LR_BARTPHO=2e-5
LR_VIT5=1e-4
MAX_LEN=256
BATCH_SIZE=16
NUM_EPOCHS=5

mkdir -p "${OUTPUT_DIR}"

echo "========================================================================"
echo "    🔬 TSSA-PRO 3-SEED VALIDATION BENCHMARK (SEEDS 42, 43, 44)"
echo "========================================================================"
echo "[*] Mục tiêu         : Đo lường phương sai ngẫu nhiên (Training Variance)"
echo "[*] Thư mục lưu riêng: ${OUTPUT_DIR}"
echo "[*] Danh sách seeds  : ${SEEDS[*]}"
echo "[*] Cấu hình chạy    : ${TARGET^^}"
echo "========================================================================"

# ------------------------------------------------------------------------------
# HÀM BỔ TRỢ: TỰ ĐỘNG IMPORT SEED 42 CÓ SẴN (NẾU ĐÃ CÓ) ĐỂ TIẾT KIỆM GPU
# ------------------------------------------------------------------------------
auto_import_existing_seed42() {
    local DEST_DIR="$1"
    local SRC_DIR="$2"

    if [ ! -f "${DEST_DIR}/eval_metrics.json" ] && [ -f "${SRC_DIR}/eval_metrics.json" ]; then
        echo "[*] Đã tìm thấy kết quả Seed 42 có sẵn tại ${SRC_DIR}. Đang copy sang ${DEST_DIR}..."
        mkdir -p "${DEST_DIR}"
        cp -r "${SRC_DIR}"/* "${DEST_DIR}/" 2>/dev/null || true
        echo "[✓] Đã import Seed 42 thành công! (Tiết kiệm ~50 phút GPU)"
    fi
}

# Auto-import Seed 42 from previous completed runs if present:
auto_import_existing_seed42 "${OUTPUT_DIR}/tssa_pro_bartpho_tay_seed42" "checkpoints/tssa_pro/tssa_tay"
auto_import_existing_seed42 "${OUTPUT_DIR}/tssa_pro_bartpho_rhade_seed42" "checkpoints/tssa_pro/tssa_rhade"
auto_import_existing_seed42 "${OUTPUT_DIR}/tssa_pro_vit5_tay_seed42" "checkpoints/tssa_pro/vit5_tssa_tay"
auto_import_existing_seed42 "${OUTPUT_DIR}/vanilla_vit5_tay_seed42" "checkpoints/vit5_vanilla_tay"
auto_import_existing_seed42 "${OUTPUT_DIR}/vanilla_bartpho_tay_seed42" "checkpoints/vanilla_tay"
auto_import_existing_seed42 "${OUTPUT_DIR}/vanilla_bartpho_rhade_seed42" "checkpoints/vanilla_rhade"

# ------------------------------------------------------------------------------
# HÀM HUẤN LUYỆN 1 MÔ HÌNH VỚI KIỂM TRA BỎ QUA NẾU ĐÃ XONG
# ------------------------------------------------------------------------------
train_single_model() {
    local EXP_NAME="$1"
    local MODEL_CKPT="$2"
    local LANG="$3"
    local LR="$4"
    local SEED="$5"
    local EXTRA_FLAGS="$6"

    local CKPT_DIR="${OUTPUT_DIR}/${EXP_NAME}"

    if [ -f "${CKPT_DIR}/eval_metrics.json" ] && [ -f "${CKPT_DIR}/test_predictions.csv" ]; then
        echo ""
        echo "[✓] [${EXP_NAME}] Đã có sẵn kết quả tại ${CKPT_DIR}. Bỏ qua!"
        return 0
    fi

    echo ""
    echo "========================================================================"
    echo ">>> BẮT ĐẦU: ${EXP_NAME} (Seed=${SEED}, LR=${LR})"
    echo "========================================================================"

    START_TIME=$(date +%s)

    python train_tssa_pro.py \
        --lang "${LANG}" \
        --model_ckpt "${MODEL_CKPT}" \
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

    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    echo "[✓] [${EXP_NAME}] Hoàn tất sau ${ELAPSED} giây."
}

# ==============================================================================
# 1. CẤU HÌNH 1: BARTPHO TÀY (tay -> vi)
# ==============================================================================
if [ "${TARGET}" == "all" ] || [ "${TARGET}" == "bartpho_tay" ]; then
    echo ""
    echo ">>> ===================================================================="
    echo ">>> CHẠY 3 SEED CHO: BARTpho Tày"
    echo ">>> ===================================================================="

    # A. Vanilla BARTpho Tày
    for S in "${SEEDS[@]}"; do
        train_single_model "vanilla_bartpho_tay_seed${S}" \
            "vinai/bartpho-syllable" "tay" "${LR_BARTPHO}" "${S}" \
            "--no_struct --no_prime --no_route --lambda_struct 0.0 --lambda_prime 0.0"
    done

    # B. Control RNG (Teacher Forward active, nhưng lambda = 0)
    for S in "${SEEDS[@]}"; do
        train_single_model "control_rng_bartpho_tay_seed${S}" \
            "vinai/bartpho-syllable" "tay" "${LR_BARTPHO}" "${S}" \
            "--use_struct --no_prime --no_route --lambda_struct 0.0 --lambda_prime 0.0 --use_centering"
    done

    # C. TSSA-Pro BARTpho Tày
    for S in "${SEEDS[@]}"; do
        train_single_model "tssa_pro_bartpho_tay_seed${S}" \
            "vinai/bartpho-syllable" "tay" "${LR_BARTPHO}" "${S}" \
            "--use_struct --use_prime --no_route --lambda_struct 0.20 --lambda_prime 0.08 --use_centering --entropy_tau 0.50 --protect_struct_fertility"
    done
fi

# ==============================================================================
# 2. CẤU HÌNH 2: BARTPHO Ê ĐÊ (rhade -> vi)
# ==============================================================================
if [ "${TARGET}" == "all" ] || [ "${TARGET}" == "bartpho_rhade" ]; then
    echo ""
    echo ">>> ===================================================================="
    echo ">>> CHẠY 3 SEED CHO: BARTpho Ê Đê"
    echo ">>> ===================================================================="

    # A. Vanilla BARTpho Ê Đê
    for S in "${SEEDS[@]}"; do
        train_single_model "vanilla_bartpho_rhade_seed${S}" \
            "vinai/bartpho-syllable" "rhade" "${LR_BARTPHO}" "${S}" \
            "--no_struct --no_prime --no_route --lambda_struct 0.0 --lambda_prime 0.0"
    done

    # B. TSSA-Pro BARTpho Ê Đê
    for S in "${SEEDS[@]}"; do
        train_single_model "tssa_pro_bartpho_rhade_seed${S}" \
            "vinai/bartpho-syllable" "rhade" "${LR_BARTPHO}" "${S}" \
            "--use_struct --use_prime --no_route --lambda_struct 0.20 --lambda_prime 0.08 --use_centering --entropy_tau 0.50 --protect_struct_fertility"
    done
fi

# ==============================================================================
# 3. CẤU HÌNH 3: ViT5 TÀY (tay -> vi)
# ==============================================================================
if [ "${TARGET}" == "all" ] || [ "${TARGET}" == "vit5_tay" ]; then
    echo ""
    echo ">>> ===================================================================="
    echo ">>> CHẠY 3 SEED CHO: ViT5 Tày"
    echo ">>> ===================================================================="

    # A. Vanilla ViT5 Tày
    for S in "${SEEDS[@]}"; do
        train_single_model "vanilla_vit5_tay_seed${S}" \
            "VietAI/vit5-base" "tay" "${LR_VIT5}" "${S}" \
            "--no_struct --no_prime --no_route --lambda_struct 0.0 --lambda_prime 0.0"
    done

    # B. TSSA-Pro ViT5 Tày
    for S in "${SEEDS[@]}"; do
        train_single_model "tssa_pro_vit5_tay_seed${S}" \
            "VietAI/vit5-base" "tay" "${LR_VIT5}" "${S}" \
            "--use_struct --use_prime --no_route --lambda_struct 0.20 --lambda_prime 0.08 --use_centering --entropy_tau 0.50 --protect_struct_fertility"
    done
fi

# ==============================================================================
# TỰ ĐỘNG BÁO CÁO KẾT QUẢ ĐA SEED & ĐÁNH GIÁ TIÊU CHÍ GO / NO-GO
# ==============================================================================
echo ""
echo "========================================================================"
echo ">>> ĐANG TỔNG HỢP VÀ TÍNH TOÁN BẢNG 3-SEED MEAN ± STD..."
echo "========================================================================"

python "${REPORT_SCRIPT}" "${OUTPUT_DIR}"

echo ""
echo "🎉 [HOÀN TẤT] Bộ thực nghiệm 3-seed đã chạy xong và xuất báo cáo đầy đủ!"
