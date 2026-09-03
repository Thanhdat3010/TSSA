#!/usr/bin/env bash
# ==============================================================================
# TSSA ABLATION RUNNER FOR RHADE & BAHNARIC
# ==============================================================================
# Chạy tự động lần lượt toàn bộ 3 biến thể bóc tách cho:
#   1. Rhade (14,969 train pairs)
#   2. Bahnaric (51,900 train pairs)
# ==============================================================================

set -e

echo "========================================================================"
echo "    🚀 BẮT ĐẦU CHẠY ABLATION STUDY CHO 2 NGÔN NGỮ: RHADE & BAHNARIC"
echo "========================================================================"

# 1. Chạy tiếng Ê Đê (Rhade)
echo ""
echo ">>> [1/2] Đang kích hoạt Ablation Study cho RHADE ..."
bash scripts/run_ablation_study.sh rhade

# 2. Chạy tiếng Ba Na (Bahnaric)
echo ""
echo ">>> [2/2] Đang kích hoạt Ablation Study cho BAHNARIC ..."
bash scripts/run_ablation_study.sh bahnaric

echo ""
echo "========================================================================"
echo "    🎉 HOÀN TẤT 100% CẢ 3 BỘ DỮ LIỆU (TÀY, RHADE, BAHNARIC)!"
echo "    Bảng tổng hợp kết quả toàn diện:"
echo "========================================================================"

python summary_results.py --ablation
