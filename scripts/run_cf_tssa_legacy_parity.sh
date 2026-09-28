#!/usr/bin/env bash
# One fixed CF configuration and its paired Vanilla run under the historical protocol.
set -euo pipefail

LANGUAGE="${1:-tay}"
SEED="${2:-42}"
if [[ "${LANGUAGE}" != "tay" || "${SEED}" != "42" ]]; then
  echo "Historical parity gate is locked to tay seed 42; no sweep." >&2
  exit 2
fi

OUT="checkpoints/fair_benchmark"
python -c 'import json; p="checkpoints/fair_benchmark/cf_tssa_pilot_tay_seed42/pilot_report.json"; d=json.load(open(p)); assert d["pilot_gate_pass"], "Pilot NO-GO"; print("[PASS] Pilot gate")'
COMMON=(--model_ckpt vinai/bartpho-syllable --lang tay --seed 42
  --selection_protocol legacy_best --num_epochs 5 --batch_size 16
  --learning_rate 2e-5 --weight_decay 0.01 --warmup_steps 500
  --max_source_length 256 --max_target_length 256 --num_beams 4
  --length_penalty 1.0 --fp16 --output_dir "${OUT}")

for METHOD in vanilla cf_tssa; do
  NAME="${METHOD}_cf_bartpho_tay_seed42_legacy_best"
  DIR="${OUT}/${NAME}"
  if python scripts/report_cf_tssa.py check --protocol legacy_best \
    --method "${METHOD}" --backbone bartpho --lang tay --seed 42 \
    --output_dir "${OUT}"; then
    echo "[SKIP] Verified complete: ${NAME}"
  elif [[ -e "${DIR}" ]]; then
    echo "[STOP] Existing incomplete/mismatched run; preserving ${DIR}" >&2
    exit 1
  else
    python train_fair_benchmark.py "${COMMON[@]}" --model_type "${METHOD}" \
      --exp_name "${NAME}"
  fi
done
python scripts/report_cf_tssa.py report --protocol legacy_best --output_dir "${OUT}"
