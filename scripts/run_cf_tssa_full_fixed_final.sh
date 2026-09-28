#!/usr/bin/env bash
# Explicitly triggered only after user reviews the historical-parity report.
set -euo pipefail

OUT="checkpoints/fair_benchmark"
for BACKBONE in bartpho vit5; do
  if [[ "${BACKBONE}" == bartpho ]]; then
    CHECKPOINT="vinai/bartpho-syllable"; LR="2e-5"; PRECISION=(--fp16)
  else
    CHECKPOINT="VietAI/vit5-base"; LR="1e-4"; PRECISION=(--bf16)
  fi
  for LANGUAGE in tay rhade bahnaric; do
    for SEED in 42 43 44; do
      COMMON=(--model_ckpt "${CHECKPOINT}" --lang "${LANGUAGE}" --seed "${SEED}"
        --selection_protocol fixed_final --num_epochs 5 --batch_size 16
        --learning_rate "${LR}" --weight_decay 0.01 --warmup_steps 500
        --max_source_length 256 --max_target_length 256 --num_beams 4
        --length_penalty 1.0 "${PRECISION[@]}" --output_dir "${OUT}")
      for METHOD in vanilla cf_tssa; do
        NAME="${METHOD}_cf_${BACKBONE}_${LANGUAGE}_seed${SEED}_fixed_final"
        DIR="${OUT}/${NAME}"
        if python scripts/report_cf_tssa.py check --protocol fixed_final \
          --method "${METHOD}" --backbone "${BACKBONE}" \
          --lang "${LANGUAGE}" --seed "${SEED}" --output_dir "${OUT}"; then
          echo "[SKIP] Verified complete: ${NAME}"
        elif [[ -e "${DIR}" ]]; then
          echo "[STOP] Existing incomplete/mismatched run; preserving ${DIR}" >&2
          exit 1
        else
          python train_fair_benchmark.py "${COMMON[@]}" --model_type "${METHOD}" \
            --exp_name "${NAME}"
        fi
      done
    done
  done
done
python scripts/report_cf_tssa.py report --protocol fixed_final --output_dir "${OUT}"
