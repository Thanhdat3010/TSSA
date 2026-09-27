#!/usr/bin/env bash
# Confirm CA-TSSA against Vanilla with the same fixed-final protocol and seeds.

set -euo pipefail

TARGET_LANG="${1:-tay}"
case "${TARGET_LANG}" in
    tay|rhade|bahnaric) ;;
    *) echo "Unsupported language: ${TARGET_LANG}" >&2; exit 2 ;;
esac

OUTPUT_DIR="checkpoints/fair_benchmark"
MODEL_CKPT="vinai/bartpho-syllable"
SEEDS=(42 43 44)

mkdir -p "${OUTPUT_DIR}"

echo "CA-TSSA fixed-final confirmation: ${TARGET_LANG} -> vi, BARTpho, seeds ${SEEDS[*]}"
echo "Locked: 5 epochs, batch 16, LR 2e-5, AdamW, weight decay 0.01, warmup 500, FP16"

for TARGET_SEED in "${SEEDS[@]}"; do
    COMMON_ARGS=(
        --model_ckpt "${MODEL_CKPT}"
        --lang "${TARGET_LANG}"
        --output_dir "${OUTPUT_DIR}"
        --selection_protocol fixed_final
        --num_epochs 5
        --batch_size 16
        --learning_rate 2e-5
        --weight_decay 0.01
        --warmup_steps 500
        --max_source_length 256
        --max_target_length 256
        --num_beams 4
        --length_penalty 1.0
        --seed "${TARGET_SEED}"
        --fp16
    )

    VANILLA_NAME="vanilla_${TARGET_LANG}_seed${TARGET_SEED}_fixed_final"
    VANILLA_DIR="${OUTPUT_DIR}/${VANILLA_NAME}"
    if [ -f "${VANILLA_DIR}/eval_metrics.json" ] && [ -f "${VANILLA_DIR}/test_predictions.csv" ]; then
        echo "[SKIP] ${VANILLA_NAME} is complete"
    else
        echo "[RUN] ${VANILLA_NAME}"
        python train_fair_benchmark.py \
            "${COMMON_ARGS[@]}" \
            --model_type vanilla \
            --exp_name "${VANILLA_NAME}"
    fi

    CA_NAME="ca_tssa_token_${TARGET_LANG}_seed${TARGET_SEED}_fixed_final"
    CA_DIR="${OUTPUT_DIR}/${CA_NAME}"
    if [ -f "${CA_DIR}/eval_metrics.json" ] && [ -f "${CA_DIR}/test_predictions.csv" ] && [ -f "${CA_DIR}/gradient_diagnostics.json" ]; then
        echo "[SKIP] ${CA_NAME} is complete"
    else
        echo "[RUN] ${CA_NAME}"
        python train_fair_benchmark.py \
            "${COMMON_ARGS[@]}" \
            --model_type ca_tssa \
            --ca_gradient_policy token_project \
            --ca_grad_cap 0.10 \
            --ca_warmup_ratio 0.10 \
            --ca_d_hidden 256 \
            --exp_name "${CA_NAME}"
    fi
done

python scripts/report_ca_tssa_fixed_final.py \
    --lang "${TARGET_LANG}" \
    --output_dir "${OUTPUT_DIR}"
