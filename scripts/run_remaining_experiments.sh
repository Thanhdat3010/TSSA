#!/usr/bin/env bash
# ==============================================================================
# Script: scripts/run_remaining_experiments.sh
# Purpose: Master automated runner for remaining TSSA evaluations & analyses.
# Usage:
#   bash scripts/run_remaining_experiments.sh            # Chạy nhanh nhiệm vụ 2, 3, 4 (~3-5 phút)
#   bash scripts/run_remaining_experiments.sh --ablation # Chạy kèm cả Ablation Study (~1.5 giờ)
# ==============================================================================

set -e

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/run_remaining_${TIMESTAMP}.log"

echo "==============================================================================" | tee -a "$MASTER_LOG"
echo "  🚀 TSSA MASTER RUNNER: TỰ ĐỘNG CHẠY TOÀN BỘ CÁC THỰC NGHIỆM CÒN LẠI" | tee -a "$MASTER_LOG"
echo "  Bắt đầu lúc: $(date)" | tee -a "$MASTER_LOG"
echo "  Ghi log chi tiết vào: $MASTER_LOG" | tee -a "$MASTER_LOG"
echo "==============================================================================" | tee -a "$MASTER_LOG"

# Kiểm tra môi trường GPU
if command -v nvidia-smi &> /dev/null; then
    echo "[*] GPU Detected: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)" | tee -a "$MASTER_LOG"
else
    echo "[!] Cảnh báo: Không tìm thấy GPU, script sẽ chạy fallback trên CPU." | tee -a "$MASTER_LOG"
fi

# ------------------------------------------------------------------------------
# BƯỚC 1: Bóc tách Độ dài câu & Câu khó (Length & Hard Instances Slicing) (~3 phút)
# ------------------------------------------------------------------------------
echo "" | tee -a "$MASTER_LOG"
echo ">>> [1/4] Đang chạy bóc tách Độ dài câu & Câu khó trên BARTpho..." | tee -a "$MASTER_LOG"
python eval_length_analysis.py \
    --checkpoints_dir checkpoints/tssa_final \
    --backbone bartpho \
    --lang all \
    --comet 2>&1 | tee -a "$MASTER_LOG"

echo "" | tee -a "$MASTER_LOG"
echo ">>> [2/4] Đang chạy bóc tách Độ dài câu & Câu khó trên ViT5..." | tee -a "$MASTER_LOG"
python eval_length_analysis.py \
    --checkpoints_dir checkpoints/tssa_final \
    --backbone vit5 \
    --lang all \
    --comet 2>&1 | tee -a "$MASTER_LOG"

# ------------------------------------------------------------------------------
# BƯỚC 2: Trích xuất Mẫu câu song ngữ định tính (Qualitative Cases) (~10 giây)
# ------------------------------------------------------------------------------
echo "" | tee -a "$MASTER_LOG"
echo ">>> [3/4] Đang trích xuất các ví dụ định tính (BARTpho & ViT5)..." | tee -a "$MASTER_LOG"
python extract_qualitative_cases.py \
    --checkpoints_dir checkpoints/tssa_final \
    --backbone bartpho 2>&1 | tee -a "$MASTER_LOG"

python extract_qualitative_cases.py \
    --checkpoints_dir checkpoints/tssa_final \
    --backbone vit5 2>&1 | tee -a "$MASTER_LOG"

# ------------------------------------------------------------------------------
# BƯỚC 3: Vẽ Ma trận Attention Heatmaps (~1 phút)
# ------------------------------------------------------------------------------
echo "" | tee -a "$MASTER_LOG"
echo ">>> [4/4] Đang vẽ Attention Heatmaps cho tiếng Ê-đê và Tày..." | tee -a "$MASTER_LOG"
python plot_attention_heatmap.py --lang rhade 2>&1 | tee -a "$MASTER_LOG" || echo "[!] Bỏ qua lỗi vẽ heatmap Rhade (nếu thiếu thư viện đồ họa)" | tee -a "$MASTER_LOG"
python plot_attention_heatmap.py --lang tay 2>&1 | tee -a "$MASTER_LOG" || echo "[!] Bỏ qua lỗi vẽ heatmap Tay" | tee -a "$MASTER_LOG"

# ------------------------------------------------------------------------------
# BƯỚC 4 (TÙY CHỌN): Ablation Study (~1.5 giờ)
# ------------------------------------------------------------------------------
if [[ "$1" == "--ablation" || "$1" == "--all" ]]; then
    echo "" | tee -a "$MASTER_LOG"
    echo ">>> [*] BẮT ĐẦU HUẤN LUYỆN ABLATION STUDY TRÊN 2 BACKBONE (Ước tính ~1.5h)..." | tee -a "$MASTER_LOG"
    bash scripts/run_unitssa_final_ablation.sh tay 2>&1 | tee -a "$MASTER_LOG"
else
    echo "" | tee -a "$MASTER_LOG"
    echo "ℹ️  Lưu ý: Ablation study (~1.5h) được bỏ qua. Nếu muốn chạy, dùng: bash scripts/run_remaining_experiments.sh --ablation" | tee -a "$MASTER_LOG"
fi

# ------------------------------------------------------------------------------
# TỔNG HỢP VÀ IN KẾT QUẢ NGAY LẬP TỨC
# ------------------------------------------------------------------------------
echo "" | tee -a "$MASTER_LOG"
echo "==============================================================================" | tee -a "$MASTER_LOG"
echo "  🎯 TỰ ĐỘNG TRÍCH XUẤT TỔNG HỢP TOÀN BỘ KẾT QUẢ BẰNG PYTHON:" | tee -a "$MASTER_LOG"
echo "==============================================================================" | tee -a "$MASTER_LOG"

python scripts/collect_experiment_results.py

echo "" | tee -a "$MASTER_LOG"
echo "✅ TẤT CẢ ĐÃ HOÀN TẤT VÀ LƯU VÀO docs/ VÀ $MASTER_LOG!" | tee -a "$MASTER_LOG"
echo "Bất cứ lúc nào bạn muốn xem lại bảng kết quả, chỉ cần gõ: python scripts/collect_experiment_results.py" | tee -a "$MASTER_LOG"
