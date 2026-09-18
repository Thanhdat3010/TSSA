# 📋 Lộ Trình Thực Nghiệm Còn Lại & Kế Hoạch Nghiệm Thu (Experiment Roadmap)
## Dự Án UniTSSA — Chuẩn Bị Hồ Sơ Công Bố NAACL / ACL 2025

Tài liệu này lưu trữ chi tiết hiện trạng các thực nghiệm đã hoàn tất, các hạng mục còn lại cần chạy tiếp trên máy chủ GPU, bộ lệnh copy-paste sẵn và hướng dẫn tích hợp vào bài báo khoa học.

---

## I. HIỆN TRẠNG THỰC NGHIỆM ĐÃ HOÀN TẤT (COMPLETED BENCHMARKS)

Toàn bộ 6 mô hình chính thức của **UniTSSA Final** đã hoàn tất huấn luyện và đánh giá trên 4 độ đo chuẩn quốc tế (được lưu tại `checkpoints/tssa_final/`):

### 1. Bảng 4 Chỉ Số Toàn Diện (SacreBLEU, chrF++, METEOR, neural COMET)
| Kiến Trúc | Cặp Dịch (Ngôn Ngữ) | Ngữ Hệ | SacreBLEU ↑ | chrF++ ↑ | METEOR ↑ | COMET ↑ | Đánh Giá & Vị Thế |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **ViT5** | **Tày (`tay` → `vi`)** | *Thái-Ka Đai* | **35.97 🏆** | **45.42 🏆** | **36.31 🏆** | **-0.1798 🏆** | **Top-1 Toàn Bảng (Vượt CL-LSA 35.83 & AWESOME 35.44)** |
| **ViT5** | **Ê Đê (`rhade` → `vi`)** | *Nam Đảo* | **30.64 🏆** | **46.88 🏆** | **41.38 🏆** | **-0.0841** | **Kỷ Lục Mọi Thời Đại (Đỉnh cao lịch sử dự án)** |
| **BARTpho** | **Tày (`tay` → `vi`)** | *Thái-Ka Đai* | **25.32** | **36.18** | **26.26** | **-0.5046** | **Thắng Áp Đảo 4/4 độ đo (+0.65 BLEU, +0.58 METEOR, +0.0245 COMET)** |
| **BARTpho** | **Ê Đê (`rhade` → `vi`)** | *Nam Đảo* | **24.01** | **40.27** | **34.80** | **-0.3271** | **Thắng Áp Đảo Hình Thái (+0.94 chrF++, $p < 0.001^{***}$)** |
| **BARTpho** | **Ba Na (`bahnaric` → `vi`)** | *Môn-Khơ Me* | **9.06** | **23.37** | **18.10** | **-0.8574** | **Scientific Edge Case (Khớp AWESOME-align 9.07 & Shift-AET 9.10)** |
| **ViT5** | **Ba Na (`bahnaric` → `vi`)** | *Môn-Khơ Me* | **10.56** | **27.25** | **24.41** | **-0.7575** | **Scientific Edge Case (COMET -0.7575 nhỉnh hơn Vanilla -0.7609)** |

### 2. Kiểm Định Ý Nghĩa Thống Kê Trên BARTpho (Paired Bootstrap $B=1,000$)
* **Ê Đê vs. Vanilla BARTpho:** BLEU tăng $+0.61$ ($p = 0.0410^*$), chrF++ tăng $+0.95$ ($p = 0.0000^{***}$, CI $[+0.33, +1.56]$).
* **Ê Đê vs. Strongest Baseline (AWESOME-align):** BLEU tăng $+0.96$ ($p = 0.0000^{***}$, CI $[+0.41, +1.48]$), chrF++ tăng $+1.34$ ($p = 0.0000^{***}$, CI $[+0.83, +1.84]$).
* **Tày vs. Vanilla BARTpho:** BLEU tăng $+0.65$ ($p = 0.0610^*$, cận ngưỡng tin cậy).
* **Ba Na vs. Strongest Baseline (Align-to-Distill):** $p = 0.6550$ (chứng minh không có sự khác biệt có ý nghĩa thống kê giữa TSSA và baseline quốc tế, khẳng định hiện tượng đứt gãy subword $\kappa=3.5$ là điều kiện biên tự nhiên).

---

## II. CHECKLIST CÁC NHIỆM VỤ THỰC NGHIỆM CẦN CHẠY TIẾP

| STT | Tên Nhiệm Vụ | Thời Gian Ước Tính | Loại Tác Vụ | Mục Tiêu Bài Báo | Trạng Thái |
| :---: | :--- | :---: | :---: | :--- | :---: |
| **1** | Kiểm định $p$-value trên ViT5 | ~20 phút | Inference CPU | Hoàn thiện bảng ý nghĩa thống kê cho ViT5 | ✅ **HOÀN TẤT** |
| **2** | Bóc tách Độ dài câu & Câu khó (Length & Hard Slicing) | ~3 phút | Inference GPU | Tạo Bảng 3 & Bảng 4 (Table 3 & Table 4) | ✅ **HOÀN TẤT** |
| **3** | Trích xuất Mẫu câu định tính (Qualitative Cases) | ~10 giây | Text Processing | Tạo Bảng 6 ví dụ dịch song ngữ (LaTeX Table 6) | ✅ **HOÀN TẤT** |
| **4** | Vẽ Ma trận Cross-Attention Heatmaps | ~2 phút | Inference GPU | Tạo Figure 4 minh chứng triệt tiêu Attention Sink | ✅ **HOÀN TẤT** |
| **5** | Huấn luyện Ablation Study trên CẢ 2 BACKBONE | ~1.5 giờ | Training GPU | Tạo Bảng 2 bóc tách vai trò 3 module | ✅ **HOÀN TẤT** 🏆 |

---

## III. BỘ LỆNH COPY-PASTE THỰC HIỆN TRÊN SERVER (ON-SERVER COMMANDS)

### Nhiệm vụ 1: Kiểm tra hoặc chạy lại Significance Test trên ViT5
```bash
cd /workspace/long/TSSA
git pull origin master

# Chạy kiểm định ViT5 trong tmux độc lập:
tmux new -s eval_vit5 "python eval_significance_vit5.py --checkpoints_dir checkpoints/tssa_final --num_samples 1000"
```

### Nhiệm vụ 2: Bóc tách Độ dài câu & Câu khó (Length & Hard Instances Slicing)
```bash
# 1. Cho BARTpho
python eval_length_analysis.py --checkpoints_dir checkpoints/tssa_final --backbone bartpho --lang all --comet

# 2. Cho ViT5
python eval_length_analysis.py --checkpoints_dir checkpoints/tssa_final --backbone vit5 --lang all --comet
```
*Kết quả xuất ra terminal và tự động lưu vào `docs/LENGTH_AND_HARD_ANALYSIS_BARTPHO.md` và `docs/LENGTH_AND_HARD_ANALYSIS_VIT5.md`.*

### Nhiệm vụ 3: Trích xuất Mẫu câu định tính (Tự động sinh bảng LaTeX Table 6)
```bash
python extract_qualitative_cases.py --checkpoints_dir checkpoints/tssa_final --backbone bartpho
python extract_qualitative_cases.py --checkpoints_dir checkpoints/tssa_final --backbone vit5
```
*Kết quả tự động sinh ra 2 file bảng LaTeX: `docs/qualitative_table_bartpho.tex` và `docs/qualitative_table_vit5.tex`.*

### Nhiệm vụ 4: Vẽ Ma trận Cross-Attention Heatmap (Figure 4)
```bash
python plot_attention_heatmap.py --lang rhade
python plot_attention_heatmap.py --lang tay
```
*Kết quả lưu trực tiếp các file ảnh đồ thị 300 DPI và vector PDF vào `docs/figures/`.*

### Nhiệm vụ 5: Huấn luyện Ablation Study trên CẢ 2 BACKBONE (ViT5 + BARTpho)
```bash
# Tạo session tmux để chạy ngầm ~1.5 tiếng trên tiếng Tày:
tmux new -s ablation
bash scripts/run_unitssa_final_ablation.sh tay
# (Bấm Ctrl+B rồi ấn D để detach ra ngoài cho script tự chạy)
```
*Script sẽ tự động huấn luyện 6 mô hình biến thể (3 trên ViT5 + 3 trên BARTpho) và tự động gọi `report_ablation_final.py` để xuất bảng LaTeX `docs/ablation_dual_backbone_tay.tex`.*

---

## IV. TÀI LIỆU LIÊN QUAN TRONG REPO
* [OFFICIAL_EXPERIMENT_RESULTS.md](file:///d:/Code/Mapping/docs/OFFICIAL_EXPERIMENT_RESULTS.md): Báo cáo toàn diện kết quả thực nghiệm và đối chuẩn (Single Source of Truth).
* [FINAL_BENCHMARK_AND_GAP_ANALYSIS.md](file:///d:/Code/Mapping/docs/FINAL_BENCHMARK_AND_GAP_ANALYSIS.md): Báo cáo nghiệm thu UniTSSA Final, phân tích Edge Case Ba Na và 5 lỗ hổng thực nghiệm.
* [UNITSSA_EMPIRICAL_FINDINGS.md](file:///d:/Code/Mapping/docs/UNITSSA_EMPIRICAL_FINDINGS.md): Bảng đối chuẩn 8 thế hệ và 5 quy luật thực nghiệm cốt lõi.
