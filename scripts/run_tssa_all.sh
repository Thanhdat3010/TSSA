#!/usr/bin/env bash
# ==============================================================================
# SCRIPT CHẠY HUẤN LUYỆN TSSA TỰ ĐỘNG CHO CẢ 3 NGÔN NGỮ QUA ĐÊM (PHIÊN BẢN MỚI)
# ==============================================================================
# Trình tự thực thi:
# 1. Tự động dọn dẹp các checkpoint TSSA cũ
# 2. Tiếng Ê Đê (rhade) -> Tự động lưu Best Model & Đánh giá
# 3. Tiếng Ba Na (bahnaric) -> Tự động lưu Best Model & Đánh giá
# 4. Tiếng Tày (tay) -> Tự động lưu Best Model & Đánh giá
# ==============================================================================

set -e

echo "========================================================================"
echo "    🚀 BẮT ĐẦU CHẠY HUẤN LUYỆN TSSA MỚI CHO CẢ 3 NGÔN NGỮ (5 EPOCHS)"
echo "========================================================================"

echo "[*] Đang dọn dẹp các checkpoint TSSA cũ..."
rm -rf checkpoints/tssa_rhade checkpoints/tssa_bahnaric checkpoints/tssa_tay

echo ""
echo ">>> [1/3] ĐANG CHẠY TSSA TRÊN TIẾNG Ê ĐÊ (rhade) ..."
python train.py --lang rhade --model_type tssa --num_epochs 5 --batch_size 16 --learning_rate 2e-5 --max_source_length 256 --max_target_length 256
echo ">>> [1/3] HOÀN TẤT TIẾNG Ê ĐÊ!"

echo ""
echo ">>> [2/3] ĐANG CHẠY TSSA TRÊN TIẾNG BA NA (bahnaric) ..."
python train.py --lang bahnaric --model_type tssa --num_epochs 5 --batch_size 16 --learning_rate 2e-5 --max_source_length 256 --max_target_length 256
echo ">>> [2/3] HOÀN TẤT TIẾNG BA NA!"

echo ""
echo ">>> [3/3] ĐANG CHẠY TSSA TRÊN TIẾNG TÀY (tay) ..."
python train.py --lang tay --model_type tssa --num_epochs 5 --batch_size 16 --learning_rate 2e-5 --max_source_length 256 --max_target_length 256
echo ">>> [3/3] HOÀN TẤT TIẾNG TÀY!"

echo ""
echo "========================================================================"
echo "    🏆 CHÚC MỪNG! ĐÃ HOÀN TẤT TOÀN BỘ 3 NGÔN NGỮ VỚI PHƯƠNG PHÁP TSSA MỚI!"
echo "========================================================================"
