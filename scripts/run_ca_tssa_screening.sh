#!/usr/bin/env bash
# CA-TSSA four-way screening under the locked fair-benchmark protocol.

set -e

TARGET_LANG="${1:-tay}"
TARGET_SEED="${2:-42}"
OUTPUT_DIR="checkpoints/fair_benchmark"
MODEL_CKPT="vinai/bartpho-syllable"
LEARNING_RATE="2e-5"
BATCH_SIZE="16"
NUM_EPOCHS="5"
MAX_LENGTH="256"

mkdir -p "${OUTPUT_DIR}"

echo "========================================================================"
echo " CA-TSSA SCREENING: ${TARGET_LANG^^} -> VI, BARTpho, seed ${TARGET_SEED}"
echo " Locked: epochs=5, batch=16, lr=2e-5, max_len=256, cap=0.10, ramp=0.10"
echo "========================================================================"

RUNS=(
    "ca_tssa_token|token_project"
    "ca_tssa_global|global_project"
    "ca_tssa_joint|joint"
    "ca_tssa_detach|detach"
)

for ITEM in "${RUNS[@]}"; do
    IFS="|" read -r METHOD_ID POLICY <<< "${ITEM}"
    EXP_NAME="${METHOD_ID}_${TARGET_LANG}_seed${TARGET_SEED}"
    CHECKPOINT_DIR="${OUTPUT_DIR}/${EXP_NAME}"

    if [ -f "${CHECKPOINT_DIR}/eval_metrics.json" ] && [ -f "${CHECKPOINT_DIR}/test_predictions.csv" ]; then
        echo ">>> [SMART SKIP] ${EXP_NAME} đã hoàn thành."
        continue
    fi

    echo ""
    echo ">>> [RUNNING] ${EXP_NAME}; policy=${POLICY}"
    python train_fair_benchmark.py \
        --model_type ca_tssa \
        --model_ckpt "${MODEL_CKPT}" \
        --lang "${TARGET_LANG}" \
        --output_dir "${OUTPUT_DIR}" \
        --exp_name "${EXP_NAME}" \
        --ca_gradient_policy "${POLICY}" \
        --ca_grad_cap 0.10 \
        --ca_warmup_ratio 0.10 \
        --ca_d_hidden 256 \
        --selection_protocol legacy_best \
        --num_epochs "${NUM_EPOCHS}" \
        --batch_size "${BATCH_SIZE}" \
        --learning_rate "${LEARNING_RATE}" \
        --max_source_length "${MAX_LENGTH}" \
        --max_target_length "${MAX_LENGTH}" \
        --seed "${TARGET_SEED}" \
        --fp16
done

python scripts/report_fair_benchmark.py "${OUTPUT_DIR}"

echo ""
echo "========================================================================"
echo " DECISION GATE — chỉ xét ca_tssa_token"
echo " Strong GO      : BLEU >= 25.84"
echo " Conditional GO : 25.40 <= BLEU < 25.84"
echo " Mechanism-only : 25.34 <= BLEU < 25.40 và diagnostics hợp lệ"
echo " NO-GO          : BLEU < 25.34 hoặc projection không kích hoạt"
echo " Không tự động chạy multi-seed và không đổi primary sau khi xem test."
echo "========================================================================"
