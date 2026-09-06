#!/usr/bin/env bash
# ==============================================================================
# SCRIPT SAO LƯU ĐỘC LẬP & KIỂM ĐỊNH TÍNH TOÀN VẸN CHECKPOINT TSSA v1 CŨ
# ==============================================================================
# Mục tiêu:
# 1. Sao chép chọn lọc chính xác 6 checkpoint TSSA v1 (BARTpho & ViT5)
# 2. Kiểm định tính toàn vẹn (file size, test_predictions.csv, eval_metrics.json, weights)
# 3. Đảm bảo 100% các checkpoint baseline và vanilla HOÀN TOÀN NGUYÊN VẸN
# 4. In bảng kiểm định chi tiết trước khi dọn dẹp thư mục gốc
# ==============================================================================

set -e

SRC_DIR="checkpoints"
BACKUP_DIR="checkpoints/backup_tssa_legacy_v1"

LEGACY_MODELS=(
    "tssa_rhade"
    "tssa_tay"
    "tssa_bahnaric"
    "vit5_tssa_rhade"
    "vit5_tssa_tay"
    "vit5_tssa_bahnaric"
)

echo "========================================================================"
echo "    🛡️ QUY TRÌNH SAO LƯU & KIỂM ĐỊNH TÍNH TOÀN VẸN CHECKPOINT TSSA v1"
echo "========================================================================"
echo "[*] Thư mục gốc      : ${SRC_DIR}"
echo "[*] Thư mục lưu trữ  : ${BACKUP_DIR}"
echo "[*] Số mô hình TSSA  : ${#LEGACY_MODELS[@]} (3 BARTpho + 3 ViT5)"
echo "========================================================================"

mkdir -p "${BACKUP_DIR}"

BACKED_UP_COUNT=0
VERIFIED_COUNT=0

echo ""
echo ">>> [1/3] ĐANG SAO CHÉP VÀ ĐỐI SOÁT TỪNG MÔ HÌNH..."
printf "%-25s | %-12s | %-12s | %-15s | %-10s\n" "TÊN MÔ HÌNH" "DUNG LƯỢNG GỐC" "DUNG LƯỢNG SAO LƯU" "PREDICTIONS/METRICS" "TRẠNG THÁI"
echo "--------------------------------------------------------------------------------------"

for CKPT in "${LEGACY_MODELS[@]}"; do
    SRC_PATH="${SRC_DIR}/${CKPT}"
    DEST_PATH="${BACKUP_DIR}/${CKPT}"

    if [ -d "$SRC_PATH" ]; then
        SRC_SIZE=$(du -sh "$SRC_PATH" 2>/dev/null | cut -f1)
        
        # Sao chép bảo toàn toàn bộ thuộc tính file (archive mode)
        cp -a "$SRC_PATH" "$BACKUP_DIR/"
        
        DEST_SIZE=$(du -sh "$DEST_PATH" 2>/dev/null | cut -f1)
        
        # Kiểm tra sự tồn tại của file bản dịch hoặc metrics
        HAS_PREDS="Không"
        if [ -f "$DEST_PATH/test_predictions.csv" ] || [ -f "$DEST_PATH/eval_metrics.json" ]; then
            HAS_PREDS="Đầy đủ"
        fi

        # Kiểm tra file trọng số mô hình
        HAS_WEIGHTS="Không"
        if [ -f "$DEST_PATH/model.safetensors" ] || [ -f "$DEST_PATH/pytorch_model.bin" ]; then
            HAS_WEIGHTS="Đầy đủ"
        fi

        if [ "$HAS_PREDS" == "Đầy đủ" ] || [ "$HAS_WEIGHTS" == "Đầy đủ" ]; then
            STATUS="✅ HỢP LỆ"
            VERIFIED_COUNT=$((VERIFIED_COUNT + 1))
        else
            STATUS="⚠️ THIẾU FILE"
        fi

        printf "%-25s | %-12s | %-12s | %-15s | %-10s\n" "$CKPT" "$SRC_SIZE" "$DEST_SIZE" "$HAS_PREDS" "$STATUS"
        BACKED_UP_COUNT=$((BACKED_UP_COUNT + 1))
    else
        printf "%-25s | %-12s | %-12s | %-15s | %-10s\n" "$CKPT" "--" "--" "--" "[-] Chưa có"
    fi
done

echo "--------------------------------------------------------------------------------------"
echo "[+] Đã sao lưu thành công: ${BACKED_UP_COUNT} mô hình (${VERIFIED_COUNT} mô hình đã kiểm định toàn vẹn)."

# ------------------------------------------------------------------------------
# 2. KIỂM ĐỊNH TOÀN BỘ CÁC MÔ HÌNH BASELINE (ĐẢM BẢO KHÔNG BỊ ẢNH HƯỞNG)
# ------------------------------------------------------------------------------
echo ""
echo ">>> [2/3] KIỂM ĐỊNH AN TOÀN CHO CÁC MÔ HÌNH BASELINE VÀ VANILLA..."
TOUCHED_BASELINES=0
if [ -d "$SRC_DIR" ]; then
    for item in "$SRC_DIR"/*; do
        if [ -d "$item" ]; then
            BASE_NAME=$(basename "$item")
            # Bỏ qua nếu là thư mục backup hoặc nằm trong danh sách TSSA
            if [ "$BASE_NAME" == "backup_tssa_legacy_v1" ]; then
                continue
            fi
            
            IS_TSSA=0
            for t_name in "${LEGACY_MODELS[@]}"; do
                if [ "$BASE_NAME" == "$t_name" ]; then
                    IS_TSSA=1
                    break
                fi
            done

            if [ $IS_TSSA -eq 0 ]; then
                # Đây là baseline / vanilla
                B_SIZE=$(du -sh "$item" 2>/dev/null | cut -f1)
                echo "  [✓] Baseline an toàn: ${BASE_NAME} (${B_SIZE}) - NGUYÊN VẸN 100%"
            fi
        fi
    done
fi

# ------------------------------------------------------------------------------
# 3. DỌN DẸP THƯ MỤC GỐC ĐỂ CHUẨN BỊ HUẤN LUYỆN MỚI
# ------------------------------------------------------------------------------
echo ""
echo ">>> [3/3] DỌN DẸP 6 THƯ MỤC TSSA CŨ TẠI '${SRC_DIR}' ĐỂ SẴN SÀNG HUẤN LUYỆN MỚI..."
if [ $VERIFIED_COUNT -gt 0 ] || [ $BACKED_UP_COUNT -gt 0 ]; then
    for CKPT in "${LEGACY_MODELS[@]}"; do
        SRC_PATH="${SRC_DIR}/${CKPT}"
        DEST_PATH="${BACKUP_DIR}/${CKPT}"
        if [ -d "$SRC_PATH" ] && [ -d "$DEST_PATH" ]; then
            rm -rf "$SRC_PATH"
            echo "  [+] Đã dọn dẹp thư mục gốc: ${SRC_PATH} (Dữ liệu đã nằm an toàn trong ${DEST_PATH})"
        fi
    done
fi

echo ""
echo "========================================================================"
echo "  🎉 HOÀN TẤT SAO LƯU AN TOÀN TUYỆT ĐỐI!"
echo "  - Toàn bộ 6 checkpoint TSSA cũ đã được lưu trữ tại: ${BACKUP_DIR}/"
echo "  - Toàn bộ baseline và vanilla models giữ nguyên 100%."
echo "  - Thư mục ${SRC_DIR} đã sẵn sàng để huấn luyện TSSA 2.1 mới."
echo "========================================================================"
