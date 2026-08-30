#!/usr/bin/env bash
# ==============================================================================
# SCRIPT ĐÁNH GIÁ NHANH CẢ 3 FILE TEST PREDICTIONS BASELINE
# ==============================================================================
set -e

echo "========================================================================"
echo "    📊 ĐANG ĐÁNH GIÁ CẢ 3 MÔ HÌNH BASELINE (Ê ĐÊ, BA NA, TÀY)"
echo "========================================================================"

echo ""
echo ">>> [1/3] Đánh giá Tiếng Ê Đê (rhade):"
python eval_checkpoint.py --predictions_csv checkpoints/bartpho_vanilla_rhade/test_predictions.csv

echo ""
echo ">>> [2/3] Đánh giá Tiếng Ba Na (bahnaric):"
python eval_checkpoint.py --predictions_csv checkpoints/bartpho_vanilla_bahnaric/test_predictions.csv

echo ""
echo ">>> [3/3] Đánh giá Tiếng Tày (tay):"
python eval_checkpoint.py --predictions_csv checkpoints/bartpho_vanilla_tay/test_predictions.csv

echo ""
echo "========================================================================"
echo "    ✅ HOÀN TẤT ĐÁNH GIÁ BASELINE!"
echo "========================================================================"
