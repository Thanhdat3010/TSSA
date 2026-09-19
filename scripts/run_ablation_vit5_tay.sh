#!/usr/bin/env bash
# ==============================================================================
# TSSA-PRO: BỘ THÍ NGHIỆM ABLATION STUDY CHUẨN MỰC TRÊN ViT5 TÀY
# ==============================================================================
# Bóc tách đóng góp khoa học của từng thành phần:
#   1. Khảo sát biên trái: lambda = 0.10
#   2. Bóc tách Centering: W/o Centering (tắt centering)
#   3. Bóc tách Cổng Entropy: W/o Dynamic Gate (tắt gate, w_s = 1.0)
#   4. Control Experiment: Teacher Shuffled (xáo trộn ngẫu nhiên vector teacher)
#
# So sánh trực tiếp với:
#   - Vanilla ViT5 Baseline: 34.99 BLEU (Đã có sẵn trong checkpoints/vit5_vanilla_tay)
#   - TSSA-Pro Full (lambda=0.20): 35.36 BLEU (Đã có sẵn trong checkpoints/tssa_sweep_tay/vit5_tay_lambda_0.2)
# ==============================================================================

set -e

NUM_EPOCHS=5
BATCH_SIZE=16
MAX_LEN=256
SEED=42
OUTPUT_DIR="checkpoints/tssa_ablation_vit5_tay"
MODEL_CKPT="VietAI/vit5-base"
LR=1e-4

mkdir -p "${OUTPUT_DIR}"

echo "========================================================================"
echo "    🔬 BẮT ĐẦU ABLATION STUDY TRÊN ViT5 TÀY (BÓC TÁCH ĐÓNG GÓP)"
echo "========================================================================"
echo "[*] Mục tiêu       : Bóc tách vai trò của Lambda, Centering, và Gate"
echo "[*] Mốc so sánh có sẵn:"
echo "    -> Vanilla ViT5 Base       : 34.99 BLEU | 44.72 chrF++"
echo "    -> TSSA-Pro Full (lam=0.2) : 35.36 BLEU | 45.12 chrF++ (+0.37)"
echo "[*] Thư mục lưu    : ${OUTPUT_DIR}"
echo "========================================================================"

run_exp() {
    local EXP_NAME="$1"
    local EXTRA_ARGS="$2"
    local DESC="$3"

    echo ""
    echo "========================================================================"
    echo ">>> [ABLATION] ${EXP_NAME}: ${DESC}"
    echo "========================================================================"

    START_TIME=$(date +%s)

    python train_tssa_pro.py \
        --lang "tay" \
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
        --sigma_kappa "0.75" \
        --use_struct \
        --use_prime \
        --no_route \
        --prime_tau "0.07" \
        --conf_threshold "0.20" \
        ${EXTRA_ARGS}

    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    echo "[✓] Hoàn tất ${EXP_NAME} sau ${ELAPSED} giây."
}

# 1. Khảo sát biên trái: lambda = 0.10 (Centering=True, Gate=True)
run_exp "vit5_tay_lambda_0.10" \
    "--lambda_struct 0.10 --lambda_prime 0.08 --use_centering --use_gate --entropy_tau 0.50" \
    "Khảo sát biên trái lambda = 0.10"

# 2. Bóc tách Centering: W/o Centering (lambda=0.20, Gate=True, TẮT Centering)
run_exp "vit5_tay_no_centering" \
    "--lambda_struct 0.20 --lambda_prime 0.08 --no_centering --use_gate --entropy_tau 0.50" \
    "Ablation: W/o Centering (Không dùng Centering để đo đóng góp của Centering)"

# 3. Bóc tách Cổng Entropy: W/o Dynamic Gate (lambda=0.20, Centering=True, TẮT Gate w_s=1.0)
run_exp "vit5_tay_no_gate" \
    "--lambda_struct 0.20 --lambda_prime 0.08 --use_centering --no_gate" \
    "Ablation: W/o Dynamic Gate (Tắt cổng entropy, w_s=1.0 cho mọi token)"

# 4. Control Experiment: Teacher Shuffled (Xáo trộn ngẫu nhiên vector Teacher)
run_exp "vit5_tay_shuffle_teacher" \
    "--lambda_struct 0.20 --lambda_prime 0.08 --use_centering --use_gate --entropy_tau 0.50 --shuffle_teacher" \
    "Control: Teacher Shuffling (Kiểm chứng ngữ nghĩa Teacher thật sự có tác dụng)"

# In bảng tổng hợp Ablation hoàn chỉnh
echo ""
python scripts/report_ablation_tay.py "${OUTPUT_DIR}"
