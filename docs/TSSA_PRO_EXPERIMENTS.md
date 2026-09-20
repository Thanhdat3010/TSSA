# 📊 Báo Cáo Thực Nghiệm TSSA-Pro & Hồ Sơ Phân Tích Kỹ Thuật (TSSA-Pro Technical & Experimental Report)

> **Tài Liệu Tổng Hợp Thực Nghiệm (Single Source of Truth - TSSA-Pro Dossier)**  
> **Dự án:** Typological Semantic-Structural Anchoring (TSSA-Pro) cho Dịch máy Ngôn ngữ Thiểu số Việt Nam  
> **Mục tiêu:** Báo cáo khách quan toàn bộ kết quả đo đạc từ GPU NVIDIA A100 cho 6 mô hình, phân tích so sánh với các baseline quốc tế, ghi nhận các hiện tượng loại hình học trên Tày và Ba Na, và thiết lập giao thức kiểm chứng đa seed (3 seeds: 42, 43, 44) theo chuẩn mực bình duyệt hội nghị quốc tế (ACL/NAACL).

---

## 📑 MỤC LỤC
1. [Tóm Tắt Kết Quả Đo Đạc Thực Tế](#1-tóm-tắt-kết-quả-đo-đạc-thực-tế)
2. [Bảng Đối Chuẩn Toàn Diện 6 Cấu Hình (Main Benchmark Suite)](#2-bảng-đối-chuẩn-toàn-diện-6-cấu-hình-main-benchmark-suite)
3. [Phân Tích So Sánh Cặp ViT5 Tày](#3-phân-tích-so-sánh-cặp-vit5-tày)
4. [Phân Tích Hiện Tượng Trên Tiếng Ba Na (Typological Boundary Condition)](#4-phân-tích-hiện-tượng-trên-tiếng-ba-na-typological-boundary-condition)
5. [Bảng Bóc Tách Thành Phần (Ablation Study Matrix - ViT5 Tày)](#5-bảng-bóc-tách-thành-phần-ablation-study-matrix---vit5-tày)
6. [Khảo Sát Dải Trọng Số $\lambda$-Sweep Trên ViT5 Tày](#6-khảo-sát-dải-trọng-số-lambda-sweep-trên-vit5-tày)
7. [Chẩn Đoán Hệ Thống & Lưu Ý Kỹ Thuật](#7-chẩn-đoán-hệ-thống--lưu-ý-kỹ-thuật)
8. [Kế Hoạch Kiểm Chứng Đa Seed (3-Seed Validation Protocol for Go/No-go)](#8-kế-hoạch-kiểm-chứng-đa-seed-3-seed-validation-protocol-for-gono-go)

---

## 1. TÓM TẮT KẾT QUẢ ĐO ĐẠC THỰC TẾ

Qua quá trình huấn luyện và đánh giá trên GPU NVIDIA A100 (5 Epochs, Batch size 16, FP16, AdamW, Seed 42), kết quả bước đầu ghi nhận:

1. **Hiệu năng trên nhóm ngôn ngữ SVO đơn lập / ít biến hình (Tày, Ê Đê):**
   * **BARTpho Ê Đê (`rhade` $\to$ `vi`):** Đạt **23.98 BLEU, 40.17 chrF++** (+0.57 BLEU so với Vanilla 23.41, +0.93 BLEU so với AWESOME-align 23.05).
   * **BARTpho Tày (`tay` $\to$ `vi`):** Đạt **25.33 BLEU, 36.11 chrF++** (+0.66 BLEU so với Vanilla 24.67, +0.13 BLEU so với AWESOME-align 25.20).
   * **ViT5 Ê Đê (`rhade` $\to$ `vi`):** Đạt **30.44 BLEU, 46.52 chrF++** (+0.16 BLEU so với Vanilla 30.28, +0.48 BLEU so với AWESOME-align 29.96).
   * **ViT5 Tày (`tay` $\to$ `vi`):** Đạt **35.31 BLEU** (lần sweep đạt 35.36 BLEU), chrF++ đạt **45.13** (+0.32 đến +0.37 BLEU so với Vanilla 34.99).

2. **Hiện tượng trên ngôn ngữ chắp dính phân mảnh cao (Ba Na):**
   * **BARTpho Ba Na (`bahnaric` $\to$ `vi`):** Đạt **9.06 BLEU, 23.48 chrF++** (-0.57 BLEU so với Vanilla 9.63).
   * **ViT5 Ba Na (`bahnaric` $\to$ `vi`):** Đạt **11.16 BLEU, 27.83 chrF++** (-0.18 BLEU so với Vanilla 11.34, chrF++ tăng +0.16).

3. **Mức chênh lệch trung bình:**  
   Mức tăng trung bình trên toàn bộ 6 ô thực nghiệm so với Vanilla là khoảng $+0.16$ BLEU. Tín hiệu thể hiện rõ nhất trên họ mô hình BARTpho (+0.57 đến +0.66 BLEU). Cần tiến hành chạy đa seed (3 seeds) để xác định độ rộng phương sai và tính vững chắc thống kê.

---

## 2. BẢNG ĐỐI CHUẨN TOÀN DIỆN 6 CẤU HÌNH (MAIN BENCHMARK SUITE)

Dữ liệu thực nghiệm đo đạc thực tế tại Seed 42:

| Kiến Trúc Backbone | Cặp Ngôn Ngữ | Số Mẫu Test | Ngữ Hệ | Vanilla Baseline | Align-to-Distill (COLING 2024) | Shift-AET (EMNLP 2020) | AWESOME-align (EACL 2021) | CL-LSA (NAACL 2021) | TSSA-Pro (Seed 42) | chrF++ (TSSA-Pro) | Δ BLEU vs Vanilla | Ghi Chú Đối Soát |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **BARTpho** | **Ê Đê (`rhade`)** | 1,000 | *Nam Đảo* | 23.41 | 22.93 | 22.38 | 23.05 | 17.85 | **23.98** | **40.17** | **+0.57** | Cao nhất trong các baseline đã đo |
| **BARTpho** | **Tày (`tay`)** | 2,295 | *Thái-Ka Đai* | 24.67 | 24.67 | 19.44 | 25.20 | 24.48 | **25.33** | **36.11** | **+0.66** | Cao nhất trong các baseline đã đo |
| **BARTpho** | **Ba Na (`bahnaric`)** | 2,001 | *Môn-Khơ Me* | 9.63 | 9.15 | 9.10 | 9.07 | 4.49 | **9.06** | **23.48** | -0.57 | Tương đương AWESOME-align (9.07) |
| **ViT5** | **Ê Đê (`rhade`)** | 1,000 | *Nam Đảo* | 30.28 | 29.34 | 29.82 | 29.96 | 27.48 | **30.44** | **46.52** | **+0.16** | Cao nhất trong các baseline đã đo |
| **ViT5** | **Tày (`tay`)** | 2,295 | *Thái-Ka Đai* | 34.99 | 33.20 | 35.14 | 35.44 | 35.83 | **35.31** | **45.13** | **+0.32** | Vượt Vanilla, thấp hơn AWESOME (-0.13) và CL-LSA (-0.52) |
| **ViT5** | **Ba Na (`bahnaric`)** | 2,001 | *Môn-Khơ Me* | 11.34 | 11.14 | 11.50 | 11.53 | 9.36 | **11.16** | **27.83** | -0.18 | chrF++ tăng +0.16; thấp hơn AWESOME (-0.37) |

---

## 3. PHÂN TÍCH SO SÁNH CẶP ViT5 TÀY

Trên cặp ViT5 Tày:
* **So với Vanilla ViT5 (34.99):** TSSA-Pro đạt **35.31** (lần sweep đạt 35.36), chênh lệch $+0.32$ đến $+0.37$ BLEU. Khoảng chênh lệch $0.05$ giữa hai lần chạy cùng cấu hình cho thấy độ rộng nhiễu tự nhiên của quá trình huấn luyện vào khoảng $\pm 0.1$ BLEU.
* **So với các Baseline:**
  * Vượt Align-to-Distill ($33.20$, $+2.11$ BLEU) và Shift-AET ($35.14$, $+0.17$ BLEU).
  * Tiệm cận AWESOME-align ($35.44$, chênh $-0.13$ BLEU, nằm trong dải nhiễu).
  * Thấp hơn CL-LSA ($35.83$, chênh $-0.52$ BLEU).
* **Đặc tính của đối thủ CL-LSA (InfoNCE):**  
  CL-LSA tối ưu hóa khoảng cách ngữ nghĩa cấp câu (sentence-level InfoNCE). Do Tày và Tiếng Việt có cấu trúc ngữ pháp tương đồng cao ($\delta = 0.06$, cùng SVO, đơn vị từ tương ứng 1-1), cơ chế kéo vector câu của InfoNCE phát huy hiệu quả cao trên riêng cặp này. Tuy nhiên, trên các cặp ngôn ngữ khác có sự sai khác về trật tự từ hoặc hình thái, CL-LSA bị suy giảm rõ rệt (BARTpho Rhade: 17.85, BARTpho Ba Na: 4.49, ViT5 Rhade: 27.48, ViT5 Ba Na: 9.36). Điều này cho thấy tính nhạy cảm cao của InfoNCE với các đặc thù loại hình học.

---

## 4. PHÂN TÍCH HIỆN TƯỢNG TRÊN TIẾNG BA NA (TYPOLOGICAL BOUNDARY CONDITION)

### 1. Hiện Trạng Số Liệu:
* Tập test Ba Na có **2,001 câu** (`data/bahnaric/test.csv`).
* Điểm BLEU của TSSA-Pro trên Ba Na là **9.06** (BARTpho) và **11.16** (ViT5), thấp hơn mô hình cơ sở Vanilla (-0.57 và -0.18 BLEU).
* Về chỉ số hình thái học chrF++, ViT5 Ba Na ghi nhận **27.83** (tăng nhẹ $+0.16$ so với Vanilla 27.67); BARTpho Ba Na đạt **23.48** (tương đương Vanilla 23.47).

### 2. So Sánh Với Các Phương Pháp Căn Chỉnh Khác Trên Ba Na:
Toàn bộ các phương pháp có can thiệp căn chỉnh biểu diễn được khảo sát trong nghiên cứu này đều ghi nhận điểm BLEU thấp hơn Vanilla trên tiếng Ba Na:
* Align-to-Distill: $9.15$ (-0.48 vs Vanilla 9.63)
* Shift-AET: $9.10$ (-0.53 vs Vanilla)
* AWESOME-align: $9.07$ (-0.56 vs Vanilla)
* CL-LSA: $4.49$ (-5.14 vs Vanilla)
* TSSA-Pro: $9.06$ (-0.57 vs Vanilla)

### 3. Nguyên Nhân Kỹ Thuật & Ngôn Ngữ Học:
* Tiếng Ba Na (ngữ hệ Môn-Khơ Me) có cấu trúc chắp dính với độ phân mảnh từ tố đo được thực tế là $\kappa \approx 3.50$ khi xử lý qua Tokenizer tiếng Việt. Một từ Ba Na bị phân tách thành nhiều subword nhỏ.
* Việc can thiệp giám sát căn chỉnh ở cấp độ subword thô khi thiếu tokenizer hình thái học chuyên dụng có thể dẫn đến hiện tượng nhiễu ranh giới từ vựng.
* Hiện tại, cơ chế suy giảm Gauss $\text{fertility\_factor}(\kappa)$ đã được áp dụng để hạ trọng số mỏ neo. Nhóm nghiên cứu sẽ kiểm tra thêm việc nhân trực tiếp hệ số này vào $\lambda_{\text{struct}}$ để đưa mức tổn hao về sát mốc Vanilla hơn.

---

## 5. BẢNG BÓC TÁCH THÀNH PHẦN (ABLATION STUDY MATRIX - ViT5 TÀY)

Được thực hiện trên `VietAI/vit5-base`, cặp Tày $\to$ Tiếng Việt, Seed 42:

| Biến Thể Mô Hình (ViT5 Tày) | Siêu Tham Số Can Thiệp | SacreBLEU ↑ | chrF++ ↑ | METEOR ↑ | COMET (wmt22) ↑ | Δ BLEU vs Vanilla (34.99) | Δ BLEU vs Full (35.36) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Vanilla Baseline** | $\lambda_{\text{struct}}=0.0, \lambda_{\text{prime}}=0.0$ | **34.99** | **44.72** | **35.93** | *[Đang đo lại]* | **Ref (0.00)** | -0.37 |
| **Ablation 1: $\lambda = 0.10$** | $\lambda_{\text{struct}}=0.10, \text{Center}, \text{Gate}$ | **35.16** | **45.01** | **36.21** | 0.7142 | **+0.17** | -0.20 |
| **TSSA-Pro Full ($\lambda^*=0.20$)** | $\lambda_{\text{struct}}=0.20, \text{Center}, \text{Gate}$ | **35.36** | **45.12** | **36.32** | 0.7164 | **+0.37** | **Ref (0.00)** |
| **Ablation 2: W/o Centering** | $\lambda_{\text{struct}}=0.20, \text{No Center}, \text{Gate}$ | **34.94** | **45.06** | **36.10** | 0.7135 | **-0.05** | **-0.42** |
| **Ablation 3: W/o Dynamic Gate** | $\lambda_{\text{struct}}=0.20, \text{Center}, w_s=1.0$ | **34.75** | **44.79** | **35.88** | 0.7108 | **-0.24** | **-0.61** |
| **Control: Teacher Shuffled** | $\lambda_{\text{struct}}=0.20, \text{Permuted Teacher}$ | **33.46** | **44.22** | **35.96** | 0.7117 | **-1.53** | **-1.90** |

> **Lưu ý kỹ thuật quan trọng:**
> 1. **Về chỉ số COMET:** Mô hình Vanilla trước đây ghi nhận điểm âm do sử dụng phiên bản `Unbabel/wmt20-comet-da`, trong khi các biến thể ablation mới được tính bằng `Unbabel/wmt22-comet-da` (cho điểm dương ~0.71). Cần chạy lại đánh giá Vanilla bằng đúng `wmt22-comet-da` để đảm bảo cùng thang đo so sánh.
> 2. **Về thử nghiệm Teacher Shuffling:** Về mặt giải tích, phép tính Barycenter là tổng trọng số Softmax theo tập hợp token, vốn bất biến với phép hoán vị nội bộ câu nếu không có padding mask. Điểm số sụp đổ xuống $33.46$ ở thử nghiệm này có sự tác động của việc xáo trộn trật tự mask padding. Để kiểm chứng chặt chẽ hơn, nhóm sẽ bổ sung thử nghiệm neo cấp câu (sentence-level mean-pool) và neo với vector ngẫu nhiên cố định.
> 3. **Về Cổng Entropy:** Cổng entropy sử dụng hàm $w_s = \exp(-\bar{H}_s / \tau_H)$ với giá trị nằm trong đoạn $[\exp(-1/\tau_H), 1.0] = [0.135, 1.0]$. Do đó, cơ chế này thực hiện giảm trọng số mềm đối với các vị trí có độ bất định cao chứ không loại bỏ hoàn toàn các token.

---

## 6. KHẢO SÁT DẢI TRỌNG SỐ $\lambda$-SWEEP TRÊN ViT5 TÀY

Dữ liệu khảo sát trên 5 điểm $\lambda \in \{0.00, 0.10, 0.20, 1.00, 5.00\}$ (ViT5 Tày, Seed 42):
* $\lambda = 0.00$ (Vanilla): $34.99$ BLEU
* $\lambda = 0.10$: $35.16$ BLEU (+0.17 vs Vanilla)
* $\lambda = 0.20$: **$35.36$ BLEU** (+0.37 vs Vanilla)
* $\lambda = 1.00$: $34.64$ BLEU (-0.35 vs Vanilla)
* $\lambda = 5.00$: $33.21$ BLEU (-1.78 vs Vanilla)

Đồ thị hình chuông đạt điểm cao nhất quanh $\lambda = 0.20$. Khi $\lambda \ge 1.00$, ràng buộc mỏ neo lớn làm suy giảm khả năng tạo sinh tự hồi quy của mô hình.

---

## 7. CHẨN ĐOÁN HỆ THỐNG & LƯU Ý KỸ THUẬT

1. **Hiệu ứng Centering:**
   * Cosine ngẫu nhiên giữa các token sau Centering đo được là `0.0168` trên BARTpho và `0.1491` trên ViT5. Centering hỗ trợ việc phân tán biểu diễn đồng đều hơn trên không gian mặt cầu đơn vị.
2. **Đo lường tỷ lệ Gradient Norm:**
   * Tỷ lệ $\|\nabla (\lambda \mathcal{L}_{\text{struct}})\| / \|\nabla \mathcal{L}_{\text{MT}}\|$ đo được tại bước khởi tạo (step 0) là $0.04\%$ tại $\lambda=0.20$ và $0.88\%$ tại $\lambda=5.00$.
   * Để đánh giá chính xác tác động động lực học, cần ghi nhận tỷ lệ này xuyên suốt các epoch huấn luyện thay vì chỉ dựa vào bước khởi tạo.

---

## 8. KẾ HOẠCH KIỂM CHỨNG ĐA SEED (3-SEED VALIDATION PROTOCOL FOR GO/NO-GO)

Để xác định chắc chắn mức độ cải thiện và loại trừ yếu tố ngẫu nhiên của seed, nhóm nghiên cứu thiết lập quy trình kiểm chứng thực nghiệm nghiêm ngặt:

### 1. Ma Trận Thử Nghiệm 3 Seeds $\times$ 3 Cặp Trọng Điểm:
* **3 Cặp khảo sát:**
  1. BARTpho Tày (`tay` $\to$ `vi`)
  2. BARTpho Ê Đê (`rhade` $\to$ `vi`)
  3. ViT5 Tày (`tay` $\to$ `vi`)
* **3 Seeds độc lập:** Seed 42, Seed 43, Seed 44.
* **Mô hình so sánh:** Cả **Vanilla** và **TSSA-Pro** đều được huấn luyện và đánh giá trên cùng tập seed và cùng giao thức đánh giá (kèm đo lại COMET wmt22 đồng nhất).

### 2. Tiêu Chí Quyết Định (Go / No-Go Decision Criteria):
* **Tiêu chí GO (Tiếp tục triển khai & viết bài công bố):**
  * Mức tăng trung bình $\Delta_{\text{mean}} \ge +0.30$ BLEU so với Vanilla trên các cặp ngôn ngữ chính.
  * Hiệu ứng mang dấu dương nhất quán trên cả 3 seed ở ít nhất 2 trong 3 cấu hình.
  * Kiểm định Paired Bootstrap Resampling ($B=1,000$) đạt mức ý nghĩa thống kê $p < 0.05$.
* **Tiêu chí NO-GO (Chuyển hướng phương pháp):**
  * Mức tăng trung bình $\Delta_{\text{mean}} \approx 0.00$ hoặc đổi dấu ngẫu nhiên giữa các seed.
  * Trong trường hợp No-Go: Chuyển hướng nghiên cứu sang phương pháp neo cấp câu thuần túy (sentence-level mean-pool) hoặc tập trung xử lý tokenization hình thái chuyên biệt cho Ba Na (như BPE-dropout).
