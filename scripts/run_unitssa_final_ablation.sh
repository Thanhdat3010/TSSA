#!/usr/bin/env bash
# ==============================================================================
# UniTSSA FINAL ABLATION STUDY: DUAL-BACKBONE COMPREHENSIVE SUITE
# ==============================================================================
# Evaluates module necessity on BOTH representative architectures:
#   1. ViT5-base  (VietAI/vit5-base, H=12, rho*=0.250, LR=1e-4) -> Record Peak Backbone (+0.98 BLEU)
#   2. BARTpho    (vinai/bartpho-syllable, H=16, rho*=0.333, LR=2e-5) -> Dense Autoregressive Backbone
#
# Primary Test Language: Tay (Tai-Kadai, 20.6K pairs, Isomorphic Baseline)
# (Optionally customizable via argument: bash scripts/run_unitssa_final_ablation.sh [lang])
#
# 3 Ablation Variants per Backbone:
#   - Variant 1: w/o Dynamic Head Routing  (--no_route --lambda_route 0.0)
#   - Variant 2: w/o Structural Anchoring  (--no_struct --lambda_struct 0.0)
#   - Variant 3: w/o Contrastive Priming   (--no_prime --lambda_prime 0.0)
#
# Baseline & Full Models are loaded from existing checkpoints:
#   - Full ViT5: checkpoints/tssa_final/vit5_tssa_${LANG} (35.97 BLEU)
#   - Vanilla ViT5: checkpoints/vit5_vanilla_${LANG} (34.99 BLEU)
#   - Full BARTpho: checkpoints/tssa_final/tssa_${LANG} (25.32 BLEU)
#   - Vanilla BARTpho: checkpoints/bartpho_vanilla_${LANG} (24.67 BLEU)
# ==============================================================================

set -e

LANG="${1:-tay}"
NUM_EPOCHS=5
BATCH_SIZE=16
MAX_LEN=256
SEED=42
OUTPUT_DIR="checkpoints/tssa_final/ablation"
REPORT_FILE="${OUTPUT_DIR}/ABLATION_REPORT.txt"

mkdir -p "${OUTPUT_DIR}"

# 2D Typological Parameters for Tay (delta=0.06, kappa=1.2)
L_STRUCT=0.20
L_PRIME=0.08
L_ROUTE=0.05
PRIME_TAU=0.07
CONF_THRESHOLD=0.20

echo "========================================================================"
echo "    🔬 UniTSSA FINAL ABLATION STUDY (DUAL-BACKBONE SUITE)"
echo "========================================================================"
echo "[*] Ngôn ngữ thực nghiệm        : ${LANG^^}"
echo "[*] Thư mục lưu trữ ablation     : ${OUTPUT_DIR}"
echo "[*] Báo cáo tự động             : ${REPORT_FILE}"
echo "[*] Siêu tham số chuẩn UniTSSA  : struct=${L_STRUCT}, prime=${L_PRIME}, route=${L_ROUTE}"
echo "========================================================================"

trap 'echo -e "\n[!] Đã nhận tín hiệu hủy (Ctrl+C). Đang dừng an toàn..."; exit 1;' INT

# ==============================================================================
# PHẦN 1: ABLATION TRÊN ViT5-base (VietAI/vit5-base, LR=1e-4, rho=0.250)
# ==============================================================================
echo ""
echo "========================================================================"
echo ">>> [KHỐI 1/2] ABLATION TRÊN ViT5-base (VietAI/vit5-base)"
echo "========================================================================"

LR_VIT5=1e-4
RHO_VIT5=0.250

# 1.1. ViT5 w/o Dynamic Head Routing (lambda_route = 0.0)
EXP_NAME="vit5_ablation_no_route_${LANG}"
CKPT_DIR="${OUTPUT_DIR}/${EXP_NAME}"
if [ -f "${CKPT_DIR}/test_predictions.csv" ]; then
    echo "[+] Đã tồn tại ${EXP_NAME}. Bỏ qua (Smart-Skip)!"
else
    echo ""
    echo ">>> [ViT5 - 1/3] Huấn luyện: ${EXP_NAME} (w/o Dynamic Head Routing)"
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
        --no_route \
        --lambda_struct "${L_STRUCT}" \
        --lambda_prime "${L_PRIME}" \
        --lambda_route 0.0 \
        --target_budget "${RHO_VIT5}" \
        --prime_tau "${PRIME_TAU}" \
        --conf_threshold "${CONF_THRESHOLD}"
fi

# 1.2. ViT5 w/o Structural Anchoring (lambda_struct = 0.0)
EXP_NAME="vit5_ablation_no_struct_${LANG}"
CKPT_DIR="${OUTPUT_DIR}/${EXP_NAME}"
if [ -f "${CKPT_DIR}/test_predictions.csv" ]; then
    echo "[+] Đã tồn tại ${EXP_NAME}. Bỏ qua (Smart-Skip)!"
else
    echo ""
    echo ">>> [ViT5 - 2/3] Huấn luyện: ${EXP_NAME} (w/o Structural Anchoring)"
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
        --no_struct \
        --use_prime \
        --use_route \
        --lambda_struct 0.0 \
        --lambda_prime "${L_PRIME}" \
        --lambda_route "${L_ROUTE}" \
        --target_budget "${RHO_VIT5}" \
        --prime_tau "${PRIME_TAU}" \
        --conf_threshold "${CONF_THRESHOLD}"
fi

# 1.3. ViT5 w/o Contrastive Priming (lambda_prime = 0.0)
EXP_NAME="vit5_ablation_no_prime_${LANG}"
CKPT_DIR="${OUTPUT_DIR}/${EXP_NAME}"
if [ -f "${CKPT_DIR}/test_predictions.csv" ]; then
    echo "[+] Đã tồn tại ${EXP_NAME}. Bỏ qua (Smart-Skip)!"
else
    echo ""
    echo ">>> [ViT5 - 3/3] Huấn luyện: ${EXP_NAME} (w/o Contrastive Priming)"
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
        --no_prime \
        --use_route \
        --lambda_struct "${L_STRUCT}" \
        --lambda_prime 0.0 \
        --lambda_route "${L_ROUTE}" \
        --target_budget "${RHO_VIT5}" \
        --prime_tau "${PRIME_TAU}" \
        --conf_threshold "${CONF_THRESHOLD}"
fi

# ==============================================================================
# PHẦN 2: ABLATION TRÊN BARTpho (vinai/bartpho-syllable, LR=2e-5, rho=0.333)
# ==============================================================================
echo ""
echo "========================================================================"
echo ">>> [KHỐI 2/2] ABLATION TRÊN BARTpho (vinai/bartpho-syllable)"
echo "========================================================================"

LR_BARTPHO=2e-5
RHO_BARTPHO=0.333

# 2.1. BARTpho w/o Dynamic Head Routing (lambda_route = 0.0)
EXP_NAME="bartpho_ablation_no_route_${LANG}"
CKPT_DIR="${OUTPUT_DIR}/${EXP_NAME}"
if [ -f "${CKPT_DIR}/test_predictions.csv" ]; then
    echo "[+] Đã tồn tại ${EXP_NAME}. Bỏ qua (Smart-Skip)!"
else
    echo ""
    echo ">>> [BARTpho - 1/3] Huấn luyện: ${EXP_NAME} (w/o Dynamic Head Routing)"
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
        --no_route \
        --lambda_struct "${L_STRUCT}" \
        --lambda_prime "${L_PRIME}" \
        --lambda_route 0.0 \
        --target_budget "${RHO_BARTPHO}" \
        --prime_tau "${PRIME_TAU}" \
        --conf_threshold "${CONF_THRESHOLD}"
fi

# 2.2. BARTpho w/o Structural Anchoring (lambda_struct = 0.0)
EXP_NAME="bartpho_ablation_no_struct_${LANG}"
CKPT_DIR="${OUTPUT_DIR}/${EXP_NAME}"
if [ -f "${CKPT_DIR}/test_predictions.csv" ]; then
    echo "[+] Đã tồn tại ${EXP_NAME}. Bỏ qua (Smart-Skip)!"
else
    echo ""
    echo ">>> [BARTpho - 2/3] Huấn luyện: ${EXP_NAME} (w/o Structural Anchoring)"
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
        --no_struct \
        --use_prime \
        --use_route \
        --lambda_struct 0.0 \
        --lambda_prime "${L_PRIME}" \
        --lambda_route "${L_ROUTE}" \
        --target_budget "${RHO_BARTPHO}" \
        --prime_tau "${PRIME_TAU}" \
        --conf_threshold "${CONF_THRESHOLD}"
fi

# 2.3. BARTpho w/o Contrastive Priming (lambda_prime = 0.0)
EXP_NAME="bartpho_ablation_no_prime_${LANG}"
CKPT_DIR="${OUTPUT_DIR}/${EXP_NAME}"
if [ -f "${CKPT_DIR}/test_predictions.csv" ]; then
    echo "[+] Đã tồn tại ${EXP_NAME}. Bỏ qua (Smart-Skip)!"
else
    echo ""
    echo ">>> [BARTpho - 3/3] Huấn luyện: ${EXP_NAME} (w/o Contrastive Priming)"
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
        --no_prime \
        --use_route \
        --lambda_struct "${L_STRUCT}" \
        --lambda_prime 0.0 \
        --lambda_route "${L_ROUTE}" \
        --target_budget "${RHO_BARTPHO}" \
        --prime_tau "${PRIME_TAU}" \
        --conf_threshold "${CONF_THRESHOLD}"
fi

echo ""
echo "========================================================================"
echo "    🎉 HOÀN TẤT TOÀN BỘ THÍ NGHIỆM ABLATION STUDY TRÊN CẢ 2 BACKBONE!"
echo "========================================================================"

# Xuất báo cáo tổng kết Ablation tự động
python report_ablation_final.py --lang "${LANG}" | tee "${REPORT_FILE}"
