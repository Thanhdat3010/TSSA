#!/usr/bin/env bash
# ==============================================================================
# TSSA ABLATION STUDY RUNNER SCRIPT ACROSS 3 MINORITY LANGUAGE DATASETS
# ==============================================================================
# Benchmarks all 3 core ablation variants across:
#   1. Rhade (Austronesian, 15.1K pairs)
#   2. Tay (Tai-Kadai, 20.6K pairs)
#   3. Bahnaric (Mon-Khmer, 51.9K pairs)
#
# Ablation Configurations:
#   - Var 1: w/o Routing Gate (--no_route) -> checks decoupling of translation heads
#   - Var 2: w/o Barycenter Struct Loss (--no_struct) -> checks token anchoring
#   - Var 3: w/o Contrastive Priming Loss (--no_prime) -> checks manifold priming
#
# Smart-Skip: Automatically detects existing evaluated checkpoints.
# Usage:
#   bash scripts/run_ablation_study.sh           # Runs all 3 languages
#   bash scripts/run_ablation_study.sh rhade     # Runs only Rhade
# ==============================================================================

set -e

NUM_EPOCHS=5
BATCH_SIZE=16
LR=2e-5
MAX_LEN=256
SEED=42

# Nếu người dùng truyền tên ngôn ngữ cụ thể, chỉ chạy ngôn ngữ đó
if [ -n "$1" ]; then
    LANGUAGES=("$1")
else
    LANGUAGES=("rhade" "tay" "bahnaric")
fi

echo "========================================================================"
echo "    BẮT ĐẦU CHẠY BỘ THÍ NGHIỆM BÓC TÁCH (ABLATION STUDY) CHO TSSA"
echo "    Danh sách ngôn ngữ: ${LANGUAGES[*]}"
echo "========================================================================"

for LANG in "${LANGUAGES[@]}"; do
    echo ""
    echo "===================================================================="
    echo ">>> BẮT ĐẦU ABLATION STUDY CHO NGÔN NGỮ: ${LANG^^}"
    echo "===================================================================="

    # --------------------------------------------------------------------------
    # 1. Biến thể 1: w/o Dynamic Head Routing (--no_route)
    # --------------------------------------------------------------------------
    EXP_NAME="tssa_no_route_${LANG}"
    PRED_FILE="checkpoints/${EXP_NAME}/test_predictions.csv"
    METRICS_FILE="checkpoints/${EXP_NAME}/eval_metrics.json"

    echo ""
    echo ">>> [1/3] [${LANG^^}] Biến thể: w/o Dynamic Head Routing (--no_route)"
    if [ -f "$PRED_FILE" ] || [ -f "$METRICS_FILE" ]; then
        echo "    [+] Đã tìm thấy kết quả của ${EXP_NAME}. BỎ QUA (Smart-Skip)!"
    else
        echo "    [*] Đang huấn luyện ${EXP_NAME} ..."
        python train.py \
            --lang "${LANG}" \
            --model_type "tssa" \
            --exp_name "${EXP_NAME}" \
            --use_struct \
            --use_prime \
            --no_route \
            --num_epochs "${NUM_EPOCHS}" \
            --batch_size "${BATCH_SIZE}" \
            --learning_rate "${LR}" \
            --max_source_length "${MAX_LEN}" \
            --max_target_length "${MAX_LEN}" \
            --seed "${SEED}" \
            --fp16
        echo "    [V] Hoàn tất: ${EXP_NAME}!"
    fi

    # --------------------------------------------------------------------------
    # 2. Biến thể 2: w/o Barycenter Struct Loss (--no_struct)
    # --------------------------------------------------------------------------
    EXP_NAME="tssa_no_struct_${LANG}"
    PRED_FILE="checkpoints/${EXP_NAME}/test_predictions.csv"
    METRICS_FILE="checkpoints/${EXP_NAME}/eval_metrics.json"

    echo ""
    echo ">>> [2/3] [${LANG^^}] Biến thể: w/o Barycenter Struct Loss (--no_struct)"
    if [ -f "$PRED_FILE" ] || [ -f "$METRICS_FILE" ]; then
        echo "    [+] Đã tìm thấy kết quả của ${EXP_NAME}. BỎ QUA (Smart-Skip)!"
    else
        echo "    [*] Đang huấn luyện ${EXP_NAME} ..."
        python train.py \
            --lang "${LANG}" \
            --model_type "tssa" \
            --exp_name "${EXP_NAME}" \
            --no_struct \
            --use_prime \
            --use_route \
            --num_epochs "${NUM_EPOCHS}" \
            --batch_size "${BATCH_SIZE}" \
            --learning_rate "${LR}" \
            --max_source_length "${MAX_LEN}" \
            --max_target_length "${MAX_LEN}" \
            --seed "${SEED}" \
            --fp16
        echo "    [V] Hoàn tất: ${EXP_NAME}!"
    fi

    # --------------------------------------------------------------------------
    # 3. Biến thể 3: w/o Sentence InfoNCE Priming Loss (--no_prime)
    # --------------------------------------------------------------------------
    EXP_NAME="tssa_no_prime_${LANG}"
    PRED_FILE="checkpoints/${EXP_NAME}/test_predictions.csv"
    METRICS_FILE="checkpoints/${EXP_NAME}/eval_metrics.json"

    echo ""
    echo ">>> [3/3] [${LANG^^}] Biến thể: w/o Sentence InfoNCE Priming (--no_prime)"
    if [ -f "$PRED_FILE" ] || [ -f "$METRICS_FILE" ]; then
        echo "    [+] Đã tìm thấy kết quả của ${EXP_NAME}. BỎ QUA (Smart-Skip)!"
    else
        echo "    [*] Đang huấn luyện ${EXP_NAME} ..."
        python train.py \
            --lang "${LANG}" \
            --model_type "tssa" \
            --exp_name "${EXP_NAME}" \
            --use_struct \
            --no_prime \
            --use_route \
            --num_epochs "${NUM_EPOCHS}" \
            --batch_size "${BATCH_SIZE}" \
            --learning_rate "${LR}" \
            --max_source_length "${MAX_LEN}" \
            --max_target_length "${MAX_LEN}" \
            --seed "${SEED}" \
            --fp16
        echo "    [V] Hoàn tất: ${EXP_NAME}!"
    fi

done

echo ""
echo "========================================================================"
echo "    [V] HOÀN TẤT TOÀN BỘ CÁC THÍ NGHIỆM ABLATION STUDY!"
echo "    Đang tiến hành tổng hợp kết quả..."
echo "========================================================================"

python summary_results.py --ablation
