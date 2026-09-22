# BẢN ĐẶC TẢ KỸ THUẬT, ĐỘNG LỰC NGHIÊN CỨU & SỐ LIỆU ĐỐI CHUẨN THỰC NGHIỆM TSSA
*(TSSA TECHNICAL SPECIFICATION, MOTIVATION, NOVELTIES & FAIR BENCHMARK DATA)*

---

## 1. ĐỘNG LỰC NGHIÊN CỨU CỦA PHƯƠNG PHÁP TSSA (RESEARCH MOTIVATION)

Phương pháp **Target-Side Semantic Anchoring (TSSA)** được thiết kế nhằm giải quyết 2 bài toán cố hữu trong Dịch máy nơ-ron ít tài nguyên (Low-Resource NMT) khi tinh chỉnh (fine-tune) mô hình ngôn ngữ lớn đã tiền huấn luyện (Pretrained Seq2Seq LM như BARTpho):

1. **Sự trôi dạt và sụp đổ biểu diễn (Representation Drift & Collapse):**
   - Khi fine-tune mô hình trên cặp ngôn ngữ thiểu số (nguồn) $\to$ tiếng Việt (đích) với tập dữ liệu nhỏ ($\approx 15,000 - 20,000$ câu), các tầng Encoder phải học biểu diễn cho các token ngôn ngữ nguồn hiếm gặp.
   - Do lượng dữ liệu giám sát nhỏ, không gian biểu diễn của Encoder ngôn ngữ nguồn dễ bị trôi dạt ra xa khỏi không gian biểu diễn tiếng Việt giàu ngữ nghĩa mà mô hình đã học từ hàng triệu câu trong giai đoạn tiền huấn luyện.
2. **Sự phân tán chú ý qua ngôn ngữ (Cross-Attention Dispersion / Attention Sinking):**
   - Decoder của mô hình Seq2Seq thường gặp khó khăn trong việc xác định các vị trí ngữ nghĩa tương ứng trên câu nguồn ít tài nguyên, dẫn đến việc các đầu chú ý (Attention Heads) tập trung vào các token đệm (`<pad>`) hoặc dấu câu thay vì các từ mang ngữ nghĩa trọng tâm.
3. **Ý tưởng cốt lõi của TSSA (Core Premise):**
   - Sử dụng chính năng lực ngữ nghĩa hoàn chỉnh của tiếng Việt (ngôn ngữ đích / high-resource language) làm **Mỏ neo Ngữ nghĩa (Semantic Anchor)**.
   - Thay vì để Encoder ngôn ngữ nguồn tự do biến đổi mà không có định hướng, TSSA thiết lập một **Giáo viên Trực tuyến (Online Frozen Teacher)** để dẫn dắt và giữ cho không gian biểu diễn của ngôn ngữ nguồn luôn bám sát không gian tiềm ẩn chuẩn của tiếng Việt.

---

## 2. CÁC TÍNH MỚI KỸ THUẬT CỐT LÕI CỦA TSSA (CORE TECHNICAL NOVELTIES)

Kiến trúc TSSA được xây dựng trên 6 đóng góp kỹ thuật độc lập:

```
[Câu Nguồn Thiểu Số X] ---> [Student Encoder f_enc] ---> [Tầng Giữa / Projector Phi] ---> [Mỏ Neo Barycenter L_struct]
                                                                                                    ^
                                                                                                    | (Stop-gradient)
[Câu Đích Tiếng Việt Y] ---> [Frozen Teacher E_T]     ---------------------------------> [Phân Phối Hậu Nghiệm A]
                                      |
                                      +------------------------------------------------> [InfoNCE Mức Câu L_prime]
```

### Tính mới 1: Giáo viên Trực tuyến Đóng băng (Online Frozen Teacher $\mathcal{E}_T$)
- Không sử dụng mô hình ngoài (như multilingual BERT hay mBART khác kiến trúc) để tránh sai lệch không gian biểu diễn và tốn kém tài nguyên.
- TSSA tận dụng chính nhánh Encoder của mô hình Seq2Seq làm Teacher và khóa cứng toàn bộ gradient:
  $$\mathcal{E}_T = \text{stop\_gradient}(f_{\text{enc}}), \quad \mathbf{h}^T = \mathcal{E}_T(\mathbf{y}) \in \mathbb{R}^{T \times d_{\text{model}}}$$
- **Ưu điểm**: Teacher và Student dùng chung 100% bộ từ vựng (Vocabulary) và Tokenizer, loại bỏ hoàn toàn độ trễ trích xuất đặc trưng ngoại tuyến và đảm bảo không có hiện tượng trôi dạt biểu diễn (Zero Representation Drift).

### Tính mới 2: Mỏ neo Trọng tâm Ngữ nghĩa Không gian Ẩn (Latent Barycentric Anchoring)
- Trong dịch máy thực tế, việc gán nhãn căn chỉnh cứng 1-1 (Hard 1-to-1 alignment) thường gây lỗi do hiện tượng phân mảnh từ tố (Subword fragmentation) và đa nghĩa.
- TSSA định nghĩa mỏ neo của token nguồn $s$ là **trọng tâm ngữ nghĩa mềm (Soft Barycenter)** của toàn bộ các token đích có liên quan:
  $$\mathbf{c}_s^T = \text{Normalize}\left( \sum_{t=1}^T \tilde{A}_{s,t} \mathbf{h}_t^T \right) \in \mathbb{S}^{D-1}$$
  trong đó ma trận $\tilde{A}_{s,t} = \frac{\exp(\langle \mathbf{h}_s^S, \mathbf{h}_t^T \rangle / \tau_{\text{align}})}{\sum_{t'} \exp(\langle \mathbf{h}_s^S, \mathbf{h}_{t'}^T \rangle / \tau_{\text{align}})}$ là phân phối căn chỉnh hậu nghiệm được chặn gradient ($\text{sg}$).

### Tính mới 3: Cổng Lọc Nhiễu Entropy Động Chuẩn Hóa Độ Dài (Length-Normalized Dynamic Entropy Gate)
- Không phải token nào cũng có cặp từ tương ứng rõ ràng trong câu đích (ví dụ: hư từ, mạo từ, trợ từ).
- TSSA tính toán độ hỗn loạn thông tin (Entropy) của phân phối căn chỉnh cho từng token nguồn $s$, chuẩn hóa theo độ dài câu đích $T$:
  $$\tilde{H}_s = \frac{-\sum_{t=1}^T \tilde{A}_{s,t} \ln \tilde{A}_{s,t}}{\ln(\max(2, T_{\text{valid}}))} \in [0, 1]$$
- Trọng số cổng điều tiết lực kéo:
  $$w_s = \exp\left(-\frac{\tilde{H}_s}{\tau_H}\right) \quad (\tau_H = 0.50)$$
  - Token có căn chỉnh sắc nét (entropy thấp $\to w_s \approx 1$): Áp dụng lực kéo mỏ neo tối đa.
  - Token phân tán mơ hồ (entropy cao $\to w_s \to 0$): Tự động triệt tiêu lực kéo, chống hiện tượng bóp méo ngữ nghĩa.

### Tính mới 4: Triệt tiêu Tính Bất đẳng hướng Không gian Biểu diễn (Batch-Centering & Hypersphere Normalization)
- Hiện tượng Anisotropy (các vector biểu diễn bị co cụm thành hình nón hẹp trong không gian) làm suy giảm độ chính xác của khoảng cách Cosine.
- TSSA áp dụng chuẩn hóa tâm theo batch trước khi chuẩn hóa L2 về mặt cầu đơn vị $\mathbb{S}^{D-1}$:
  $$\tilde{\mathbf{h}} = \frac{\mathbf{h} - \boldsymbol{\mu}_{\text{batch}}}{\|\mathbf{h} - \boldsymbol{\mu}_{\text{batch}}\|_2}$$

### Tính mới 5: Hiệu chỉnh Độ nở Từ tố Thích ứng (Subword Fertility Attenuation $\kappa$)
- Các ngôn ngữ có cấu trúc từ tố phức tạp thường bị tách thành nhiều mảnh subwords (độ nở từ tố $\kappa = \frac{\text{total subwords}}{\text{total words}} > 1.0$).
- TSSA tích hợp hệ số suy giảm Gaussian liên tục để điều tiết hàm InfoNCE:
  $$\gamma(\kappa) = \exp\left( -\frac{(\max(1.0, \kappa) - 1.0)^2}{2 \sigma_\kappa^2} \right) \quad (\sigma_\kappa = 0.75)$$

### Tính mới 6: Bộ Chiếu Phi Tuyến Độc Lập ở Tầng Giữa (Decoupled Middle-Layer Projection - TSSA V4)
- Can thiệp tại tầng giữa của Encoder (Layer 3 trên 6 tầng của BARTpho) thay vì tầng cuối (Layer 6).
- Sử dụng mạng chiếu 2 tầng MLP Projector ($\mathbb{R}^{1024} \to \mathbb{R}^{2048} \to \mathbb{R}^{1024}$ + GELU + LayerNorm) kết hợp hàng đợi bộ nhớ FIFO (MoCo Queue $Q=256$) để tách biệt gradient căn chỉnh ngữ nghĩa khỏi các tầng trên (Layer 4--6) dành cho cú pháp dịch mã.

---

## 3. THÔNG SỐ TẬP DỮ LIỆU & PHẦN CỨNG (DATASET & HARDWARE SPECIFICATIONS)

### 3.1. Tập Dữ Liệu Thực Nghiệm (Dataset)
- **Cặp ngôn ngữ**: Tiếng Tày sang Tiếng Việt (`tay` $\to$ `vi`).
- **Phân chia tập dữ liệu (Splits)**:
  - Tập huấn luyện (`train.csv`): **20,554** cặp câu song ngữ.
  - Tập kiểm tra (`test.csv`): **2,295** cặp câu song ngữ.
- **Độ dài chuỗi tối đa**: Source = 256 tokens \| Target = 256 tokens.
- **Tokenizer**: `vinai/bartpho-syllable` (Syllable-level BPE, từ điển 40,000 tokens).
- **Fertility Rate (Tày)**: 1.20 tokens/từ.

### 3.2. Môi Trường Thực Thi Phần Cứng (Hardware Environment)
- **GPU**: NVIDIA A100 Tensor Core GPU.
- **Hệ điều hành**: Linux Ubuntu 22.04 LTS, PyTorch 2.x, Transformers 4.49+, CUDA 12.x.
- **Kiểm soát tính ngẫu nhiên (RNG Seed Control)**:
  - Seed đối chuẩn: `seed = 42` (thiết lập đồng thời trên Python, NumPy, PyTorch, CUDA, Transformers).
  - DataLoader: `data_seed = 42`.

---

## 4. KIẾN TRÚC MÔ HÌNH NỀN & SIÊU THAM SỐ CHUNG (SHARED BACKBONE & HYPERPARAMETERS)

| Siêu tham số | Giá trị quy chuẩn |
|---|---|
| **Mô hình nền (Backbone)** | `vinai/bartpho-syllable` (BARTpho-base, 135M tham số) |
| **Số tầng Encoder / Decoder** | 6 Encoder layers / 6 Decoder layers |
| **Kích thước ẩn ($d_{model}$)** | 1,024 |
| **Kích thước Feed-Forward ($d_{ff}$)** | 4,096 |
| **Số Attention Heads** | 16 heads ($d_k = 64$) |
| **Batch Size** | 16 (per-device train và eval) |
| **Optimizer** | AdamW ($\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$, weight decay = 0.01) |
| **Learning Rate** | $2 \times 10^{-5}$ cho toàn bộ mô hình |
| **Lập lịch LR** | Linear Warmup 500 steps $\to$ Linear Decay về 0 |
| **Số Epochs** | 5 epochs (Early stopping: patience = 3 epochs theo SacreBLEU test) |
| **Precision** | Mixed Precision FP16 |
| **Cấu hình giải mã** | Beam Search, `num_beams = 4`, `length_penalty = 1.0`, max len = 256 |
| **Thang đo định lượng** | SacreBLEU (exp smoothing), chrF++ (word_order=2), METEOR, COMET (`wmt22-comet-da`) |

---

## 5. THÔNG SỐ VÀ CÔNG THỨC CỦA 9 HỆ THỐNG ĐÃ ĐỐI CHUẨN

1. **Vanilla Baseline**:
   - Vị trí can thiệp: Không có.
   - Hàm mục tiêu: $L_{total} = L_{MT}$ (Cross-Entropy tiêu chuẩn).
2. **AWESOME-align (EACL 2021)**:
   - Vị trí can thiệp: Encoder Layer 6 (Output).
   - Hàm mục tiêu: $L_{total} = L_{MT} + 0.10 \times L_{awesome}$ (16 heads, dimension 1,024).
3. **CL-LSA (NAACL 2021)**:
   - Vị trí can thiệp: Encoder Layer 6 (Output).
   - Hàm mục tiêu: $L_{total} = L_{MT} + 0.10 \times L_{InfoNCE}$ (In-batch negatives $B-1=15$, $\tau = 0.07$).
4. **Align-to-Distill (COLING 2024)**:
   - Vị trí can thiệp: Encoder Layer 6 (Output).
   - Hàm mục tiêu: $L_{total} = L_{MT} + 0.10 \times D_{KL}(A_{student} \parallel A_{teacher})$ (Chưng cất Cross-Attention, $\tau = 0.10$).
5. **Shift-AET (EMNLP 2020)**:
   - Vị trí can thiệp: Encoder Layer 6 (Output).
   - Hàm mục tiêu: $L_{total} = L_{MT} + 0.10 \times L_{AET}$ (Autoencoding tái tạo + tịnh tiến tự hồi quy).
6. **TSSA-Pro (Bản Mỏ Neo Cũ)**:
   - Vị trí can thiệp: Encoder Layer 6 (Output qua phép chiếu tuyến tính $1,024 \to 1,024$).
   - Hàm mục tiêu: $L_{total} = L_{MT} + 0.20 \times L_{Barycenter} + 0.08 \times L_{InfoNCE}$.
   - Tham số: $\tau_H = 0.50$, $\tau_{prime} = 0.07$, confidence threshold = 0.20, $\sigma_\kappa = 0.75$.
7. **TSSA-V4 (Sentence InfoNCE)**:
   - Vị trí can thiệp: Encoder Layer 3 (Tầng giữa $6 // 2 = 3$ qua 2-tầng Decoupled MLP Projector).
   - Hàm mục tiêu: $L_{total} = L_{MT} + 0.10 \times L_{InfoNCE}^{Queue}$.
   - Tham số: MoCo FIFO Memory Queue $Q = 256$ vector đích, $\tau = 0.07$.
8. **TSSA-V4 (Token Barycenter)**:
   - Vị trí can thiệp: Encoder Layer 3 (Tầng giữa $6 // 2 = 3$ qua 2-tầng Decoupled MLP Projector).
   - Hàm mục tiêu: $L_{total} = L_{MT} + 0.10 \times L_{Barycenter}^{Gate}$.
   - Tham số: Soft Barycenter + Dynamic Entropy Gate ($\tau_H = 0.50$, $\tau_{align} = 0.10$).
9. **TSSA-V4 (Hybrid Dual-Level)**:
   - Vị trí can thiệp: Encoder Layer 3 qua 2-tầng Decoupled MLP Projector.
   - Hàm mục tiêu: $L_{total} = L_{MT} + 0.08 \times L_{InfoNCE}^{Queue} + 0.05 \times L_{Barycenter}^{Gate}$.
   - Tham số: Hàng đợi $Q = 256$, $\tau = 0.07$, $\tau_H = 0.50$, $\tau_{align} = 0.10$.

---

## 6. BẢNG SỐ LIỆU THỰC NGHIỆM ĐỐI CHUẨN ĐẦY ĐỦ (OFFICIAL BENCHMARK RESULTS TABLE)

*Toàn bộ số liệu dưới đây được đo đạc trực tiếp trên tập kiểm tra gồm 2,295 câu của Tiếng Tày (`tay` $\to$ `vi`), Seed 42, Beam 4.*

| STT | Hệ Thống / Mô Hình | Vị trí can thiệp | SacreBLEU ↑ | chrF++ ↑ | METEOR ↑ | COMET ↑ | $\Delta$ BLEU vs Vanilla | $\Delta$ chrF++ vs Vanilla |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **AWESOME-align (EACL 2021)** | Encoder Layer 6 | **25.39** | **36.30** | **26.41** | **0.6579** | $+0.05$ | $+0.25$ |
| 2 | **TSSA-Pro (Old Anchor)** | Encoder Layer 6 | 25.36 | 36.00 | 25.88 | 0.6539 | $+0.02$ | $-0.05$ |
| 3 | **Vanilla Baseline (BARTpho)** | **Không can thiệp** | **25.34** | **36.05** | **26.12** | **0.6502** | **Ref (0.00)** | **Ref (0.00)** |
| 4 | **TSSA-V4 (Token Barycenter)** | Encoder Layer 3 | 24.83 | 35.93 | 26.10 | 0.6555 | $-0.51$ | $-0.12$ |
| 5 | **TSSA-V4 (Sentence InfoNCE)** | Encoder Layer 3 | 24.65 | 35.34 | 25.50 | 0.6502 | $-0.69$ | $-0.71$ |
| 6 | **Shift-AET (EMNLP 2020)** | Encoder Layer 6 | 24.52 | 35.60 | 25.95 | 0.6525 | $-0.82$ | $-0.45$ |
| 7 | **CL-LSA (NAACL 2021)** | Encoder Layer 6 | 24.45 | 34.92 | 25.34 | 0.6458 | $-0.89$ | $-1.13$ |
| 8 | **Align-to-Distill (COLING 2024)** | Encoder Layer 6 | 23.34 | 33.93 | 23.93 | 0.6410 | $-2.00$ | $-2.12$ |
| 9 | **TSSA-V4 (Hybrid Dual-Level)** | Encoder Layer 3 | 22.18 | 32.84 | 23.06 | 0.6329 | $-3.16$ | $-3.21$ |

---

## 7. CÁC MỐC CHUẨN VÀ RÀNG BUỘC CHO PHƯƠNG PHÁP MỚI (TARGET BENCHMARKS & CONSTRAINTS)

1. **Mốc chuẩn cơ sở cần vượt qua**:
   - Mốc Vanilla Baseline: **25.34 BLEU** (chrF++: 36.05, COMET: 0.6502).
   - Mốc Baseline tốt nhất hiện có (AWESOME-align): **25.39 BLEU** (chrF++: 36.30, COMET: 0.6579).
2. **Ngưỡng gia tăng có ý nghĩa thống kê**:
   - Để vượt qua dải nhiễu ngẫu nhiên của tập test (paired bootstrap confidence interval $\approx \pm 0.35$ BLEU), phương pháp mới cần đạt:
     $$\text{SacreBLEU} \ge \mathbf{25.75 - 25.85} \quad (\Delta \text{BLEU} \ge \mathbf{+0.40 \sim +0.50})$$
3. **Các tham số bắt buộc giữ nguyên để đối chuẩn công bằng**:
   - Dữ liệu: Cùng tập `train.csv` (20,554 mẫu) và `test.csv` (2,295 mẫu) của `data_processed/tay`.
   - Mô hình nền: `vinai/bartpho-syllable`.
   - Siêu tham số nền: Batch size 16, Max length 256, Epochs 5, Optimizer AdamW với learning rate $2 \times 10^{-5}$, FP16.
   - Seed đối chứng: Seed 42.
   - Cấu hình suy luận: Beam search 4, length penalty 1.0.
