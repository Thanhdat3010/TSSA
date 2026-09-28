#!/usr/bin/env bash
set -euo pipefail

python scripts/run_cf_tssa_stage1.py --lang tay --seed 42 --steps 500 \
  --model_ckpt vinai/bartpho-syllable \
  --output_dir checkpoints/fair_benchmark/cf_tssa_pilot_tay_seed42
