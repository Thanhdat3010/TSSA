#!/usr/bin/env bash
# ==============================================================================
# UNIFIED ABLATION & CAUSAL STUDY RUNNER SCRIPT
# ==============================================================================
# Chạy toàn bộ các thí nghiệm bóc tách thành phần (2^3 Components),
# Thẩm định nhân quả Causal Head Pruning, và Robustness Noise.
# ==============================================================================

set -e

NUM_EPOCHS=5
BATCH_SIZE=16
LR=2e-5
MAX_LEN=256
SEED=42
LANG="rhade"

echo "========================================================================"
echo "    BẮT ĐẦU CHẠY CÁC BÀI ABLATION TRÊN NGÔN NGỮ: ${LANG^^}"
echo "========================================================================"

# --- 1. Ablation 1: Ma trận 2^3 Bóc tách thành phần ---
echo ">>> [1/4] Chạy Ablation 1: Bóc tách từng thành phần Loss..."

# A1: Base NMT (L_MT) -> chính là bartpho_vanilla
# A2: Struct Only (L_struct)
python train.py --lang "${LANG}" --model_type tssa --use_struct --no_prime --no_route --num_epochs "${NUM_EPOCHS}" --batch_size "${BATCH_SIZE}" --output_dir "checkpoints/ablation_struct_only"

# A3: Prime Only (L_prime)
python train.py --lang "${LANG}" --model_type tssa --no_struct --use_prime --no_route --num_epochs "${NUM_EPOCHS}" --batch_size "${BATCH_SIZE}" --output_dir "checkpoints/ablation_prime_only"

# A4: Route Only (L_route)
python train.py --lang "${LANG}" --model_type tssa --no_struct --no_prime --use_route --num_epochs "${NUM_EPOCHS}" --batch_size "${BATCH_SIZE}" --output_dir "checkpoints/ablation_route_only"

# A5: Struct + Prime
python train.py --lang "${LANG}" --model_type tssa --use_struct --use_prime --no_route --num_epochs "${NUM_EPOCHS}" --batch_size "${BATCH_SIZE}" --output_dir "checkpoints/ablation_struct_prime"

# A6: Struct + Route
python train.py --lang "${LANG}" --model_type tssa --use_struct --no_prime --use_route --num_epochs "${NUM_EPOCHS}" --batch_size "${BATCH_SIZE}" --output_dir "checkpoints/ablation_struct_route"

# A7: Prime + Route
python train.py --lang "${LANG}" --model_type tssa --no_struct --use_prime --use_route --num_epochs "${NUM_EPOCHS}" --batch_size "${BATCH_SIZE}" --output_dir "checkpoints/ablation_prime_route"

# A8: Full TSSA (Struct + Prime + Route)
python train.py --lang "${LANG}" --model_type tssa --use_struct --use_prime --use_route --num_epochs "${NUM_EPOCHS}" --batch_size "${BATCH_SIZE}" --output_dir "checkpoints/ablation_full_tssa"

# --- 2. Ablation 3: Causal Head Pruning ---
echo ">>> [2/4] Chạy Ablation 3: Causal Head Pruning (Top-K vs Random-K vs Bottom-K)..."
python evaluate.py --checkpoint_dir "checkpoints/ablation_full_tssa/tssa_rhade" --lang "${LANG}" --run_causal_pruning

# --- 3. Ablation 4: Robustness trước Nhiễu ---
echo ">>> [3/4] Chạy Ablation 4: Độ bền vững trước nhiễu chính tả / xóa dấu..."
python evaluate.py --checkpoint_dir "checkpoints/ablation_full_tssa/tssa_rhade" --lang "${LANG}" --run_robustness

echo "========================================================================"
echo "    HOÀN TẤT TOÀN BỘ CÁC BÀI THÍ NGHIỆM ABLATION!"
echo "========================================================================"
