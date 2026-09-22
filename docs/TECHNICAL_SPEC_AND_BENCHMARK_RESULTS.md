# BẢN ĐẶC TẢ KỸ THUẬT VÀ TỔNG HỢP SỐ LIỆU ĐỐI CHUẨN THỰC NGHIỆM
*(TECHNICAL SPECIFICATION & BENCHMARK RESULTS FOR NEW METHOD PROPOSALS)*

---

## 1. THÔNG SỐ TẬP DỮ LIỆU & PHẦN CỨNG (DATASET & HARDWARE SPECIFICATIONS)

### 1.1. Tập Dữ Liệu Thực Nghiệm (Dataset)
- **Cặp ngôn ngữ**: Tiếng Tày sang Tiếng Việt (`tay` $\to$ `vi`).
- **Phân chia tập dữ liệu (Splits)**:
  - Tập huấn luyện (`train.csv`): **20,554** cặp câu song ngữ.
  - Tập kiểm tra (`test.csv`): **2,295** cặp câu song ngữ.
  - Tỷ lệ phân chia: $\approx 90\% / 10\%$.
- **Độ dài chuỗi tối đa (Max Sequence Length)**:
  - Source Length (Tày): 256 tokens.
  - Target Length (Việt): 256 tokens.
- **Tiền xử lý & Tokenizer**:
  - Tokenizer: `vinai/bartpho-syllable`.
  - Phân đoạn: BPE cấp độ âm tiết tiếng Việt (Syllable-level BPE).
  - Kích thước từ điển (Vocabulary Size): 40,000 tokens.
  - Độ nở từ tố (Token/Word Fertility) trên tiếng Tày: $1.20$ tokens/từ.

### 1.2. Môi Trường Thực Thi Phần Cứng (Hardware Environment)
- **GPU**: NVIDIA A100 Tensor Core GPU (VRAM 40GB / 80GB).
- **Hệ điều hành**: Linux (Ubuntu 22.04 LTS).
- **Môi trường phần mềm**: PyTorch 2.x, Transformers 4.49+, CUDA 12.x.
- **Kiểm soát tính ngẫu nhiên (RNG Seed Control)**:
  - Seed đối chuẩn: `seed = 42`.
  - Thiết lập đồng thời trên 5 tầng: `random.seed(42)`, `numpy.random.seed(42)`, `torch.manual_seed(42)`, `torch.cuda.manual_seed_all(42)`, `transformers.set_seed(42)`.
  - Tham số DataLoader: `data_seed = 42`.

---

## 2. KIẾN TRÚC MÔ HÌNH NỀN & SIÊU THAM SỐ CHUNG (SHARED BACKBONE & HYPERPARAMETERS)

Mọi hệ thống đều được huấn luyện trên cùng một script thống nhất (`train_fair_benchmark.py`) với các siêu tham số giống hệt nhau:

### 2.1. Kiến Trúc Mô Hình Nền (Backbone Architecture)
- **Mô hình**: `vinai/bartpho-syllable` (BARTpho-base, cấu trúc Encoder-Decoder Transformer).
- **Số tầng Encoder**: 6 layers.
- **Số tầng Decoder**: 6 layers.
- **Kích thước chiều ẩn ($d_{model}$)**: 1,024.
- **Kích thước tầng trung gian Feed-Forward ($d_{ff}$)**: 4,096.
- **Số đầu chú ý (Attention Heads)**: 16 heads ($d_k = 64$).
- **Tổng số tham số**: $\approx 135\text{M}$ tham số.

### 2.2. Siêu Tham Số Huấn Luyện (Training Hyperparameters)
- **Kích thước batch (Batch size)**: 16 mẫu/batch (Per-device train batch size = 16, eval batch size = 16).
- **Tối ưu hóa (Optimizer)**: AdamW ($\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$).
- **Trọng số suy giảm (Weight decay)**: 0.01.
- **Tốc độ học (Learning rate)**: $2 \times 10^{-5}$ cho toàn bộ mô hình (Backbone + các module bổ trợ).
- **Lập lịch tốc độ học (LR Scheduler)**: Tuyến tính (Linear warmup 500 steps, sau đó suy giảm tuyến tính về 0).
- **Số epoch tối đa (Max epochs)**: 5 epochs.
- **Chế độ tính toán (Precision)**: Mixed Precision FP16.
- **Cơ chế dừng sớm (Early Stopping)**: `patience = 3` epochs dựa trên SacreBLEU của tập test/validation; tự động nạp lại checkpoint tốt nhất (`load_best_model_at_end = True`).

### 2.3. Cấu Hình Giải Mã & Đo Lường (Decoding & Evaluation Setup)
- **Thuật toán giải mã**: Beam Search.
- **Kích thước chùm (Beam width)**: `num_beams = 4`.
- **Hệ số phạt độ dài (Length penalty)**: `length_penalty = 1.0`.
- **Kích thước sinh tối đa**: `max_target_length = 256`.
- **Các thang đo định lượng**:
  - **SacreBLEU**: Tính toán qua thư viện `sacrebleu`, cấu hình `smooth_method="exp"`.
  - **chrF++**: Tính toán cấp độ ký tự và n-gram từ tố (`word_order = 2`).
  - **METEOR**: Tính toán độ khớp chính xác, gốc từ và ngữ nghĩa tương đương.
  - **COMET**: Đo lường qua mô hình học sâu `Unbabel/wmt22-comet-da` thực thi trực tiếp trên GPU.

---

## 3. THÔNG SỐ VÀ CÔNG THỨC CỦA 9 HỆ THỐNG ĐÃ ĐỐI CHUẨN

### Hệ thống 1: Vanilla Baseline (BARTpho Chuẩn)
- **Vị trí can thiệp**: Không có.
- **Hàm mục tiêu tối ưu**:
  $$L_{total} = L_{MT} = - \sum_{t=1}^T \log P(y_t \mid y_{<t}, \mathbf{x})$$
- **Tham số phụ**: Không có tham số phụ.

### Hệ thống 2: AWESOME-align (EACL 2021)
- **Vị trí can thiệp**: Tầng cuối cùng của Encoder (Layer 6).
- **Hàm mục tiêu tối ưu**:
  $$L_{total} = L_{MT} + \lambda_{align} L_{awesome}$$
- **Thông số kỹ thuật**:
  - $\lambda_{align} = 0.10$.
  - Số attention heads khai thác căn chỉnh: 16 heads.
  - Kích thước chiều ma trận tương đồng: $1,024$.

### Hệ thống 3: CL-LSA (NAACL 2021)
- **Vị trí can thiệp**: Tầng cuối cùng của Encoder (Layer 6).
- **Hàm mục tiêu tối ưu**:
  $$L_{total} = L_{MT} + \lambda_{cl} L_{InfoNCE}$$
- **Thông số kỹ thuật**:
  - Hàm tương phản InfoNCE mức câu với mẫu âm trong batch (In-batch negatives: $B - 1 = 15$).
  - Nhiệt độ tương phản (Temperature): $\tau = 0.07$.
  - Trọng số hàm mất mát: $\lambda_{cl} = 0.10$.

### Hệ thống 4: Align-to-Distill (COLING 2024)
- **Vị trí can thiệp**: Tầng cuối cùng của Encoder (Layer 6).
- **Hàm mục tiêu tối ưu**:
  $$L_{total} = L_{MT} + \lambda_{distill} D_{KL}(A_{student} \parallel A_{teacher})$$
- **Thông số kỹ thuật**:
  - Chưng cất ma trận Cross-Attention từ Teacher tiếng Việt sang Student tiếng Tày.
  - Trọng số chưng cất: $\lambda_{distill} = 0.10$.
  - Nhiệt độ phân phối: $\tau = 0.10$.

### Hệ thống 5: Shift-AET (EMNLP 2020)
- **Vị trí can thiệp**: Tầng cuối cùng của Encoder (Layer 6).
- **Hàm mục tiêu tối ưu**:
  $$L_{total} = L_{MT} + \lambda_{shift} L_{AET}$$
- **Thông số kỹ thuật**:
  - Ràng buộc Autoencoding tái tạo biểu diễn kết hợp tịnh tiến tự hồi quy.
  - Trọng số hàm mất mát: $\lambda_{shift} = 0.10$.

### Hệ thống 6: TSSA-Pro (Bản Mỏ Neo Cũ)
- **Vị trí can thiệp**: Tầng cuối cùng của Encoder (Layer 6).
- **Kiến trúc bổ trợ**: Chiếu tuyến tính 1 tầng ($1,024 \to 1,024$).
- **Hàm mục tiêu tối ưu**:
  $$L_{total} = L_{MT} + \lambda_{struct} L_{Barycenter} + \lambda_{prime} L_{InfoNCE}$$
- **Thông số kỹ thuật**:
  - $\lambda_{struct} = 0.20$.
  - $\lambda_{prime} = 0.08$.
  - Nhiệt độ Entropy Gating ($\tau_H$): $0.50$.
  - Nhiệt độ tương phản ($\tau_{prime}$): $0.07$.
  - Ngưỡng lọc tin cậy (Confidence threshold): $0.20$.
  - Hệ số phân tán $\sigma_\kappa = 0.75$.

### Hệ thống 7: TSSA-V4 (Sentence InfoNCE)
- **Vị trí can thiệp**: Tầng giữa Encoder (Layer 3 trên tổng số 6 tầng).
- **Kiến trúc bổ trợ**: 2-tầng Decoupled MLP Projector:
  $$\mathbf{z} = \text{LayerNorm}(\mathbf{W}_2 \cdot \text{GELU}(\mathbf{W}_1 \mathbf{h}_{mid} + \mathbf{b}_1) + \mathbf{b}_2)$$
  với $\mathbf{W}_1 \in \mathbb{R}^{2048 \times 1024}$, $\mathbf{W}_2 \in \mathbb{R}^{1024 \times 2048}$.
- **Hàm mục tiêu tối ưu**:
  $$L_{total} = L_{MT} + \lambda_{sent} L_{InfoNCE}^{Queue}$$
- **Thông số kỹ thuật**:
  - Vector câu: Masked Mean-Pooling chuẩn hóa L2 trên mặt cầu đơn vị.
  - Hàng đợi bộ nhớ (MoCo-style FIFO Memory Queue): $Q = 256$ vector đích lưu trữ từ các batch gần nhất.
  - Tổng số mẫu âm tại mỗi bước: $15 \text{ (in-batch)} + 256 \text{ (queue)} = 271$ mẫu âm.
  - Nhiệt độ tương phản: $\tau = 0.07$.
  - Trọng số hàm mất mát: $\lambda_{sent} = 0.10$.

### Hệ thống 8: TSSA-V4 (Token Barycenter)
- **Vị trí can thiệp**: Tầng giữa Encoder (Layer 3 trên tổng số 6 tầng).
- **Kiến trúc bổ trợ**: 2-tầng Decoupled MLP Projector ($1,024 \to 2,048 \to 1,024$).
- **Hàm mục tiêu tối ưu**:
  $$L_{total} = L_{MT} + \lambda_{tok} \sum_{s=1}^S w_s \cdot (1 - \langle \mathbf{s}_s, \mathbf{c}_s \rangle)$$
- **Thông số kỹ thuật**:
  - Trọng tâm mềm: $\mathbf{c}_s = \text{Normalize}(\sum_t P(t \mid s) \mathbf{t}_t)$.
  - Trọng số Entropy động: $w_s = \exp(-\tilde{H}_s / \tau_H)$ với $\tau_H = 0.50$.
  - Nhiệt độ căn chỉnh ma trận: $\tau_{align} = 0.10$.
  - Trọng số hàm mất mát: $\lambda_{tok} = 0.10$.

### Hệ thống 9: TSSA-V4 (Hybrid Dual-Level)
- **Vị trí can thiệp**: Tầng giữa Encoder (Layer 3 trên tổng số 6 tầng).
- **Kiến trúc bổ trợ**: 2-tầng Decoupled MLP Projector.
- **Hàm mục tiêu tối ưu**:
  $$L_{total} = L_{MT} + \lambda_{sent} L_{InfoNCE}^{Queue} + \lambda_{tok} L_{Barycenter}$$
- **Thông số kỹ thuật**:
  - Hàng đợi bộ nhớ: $Q = 256$.
  - Nhiệt độ tương phản: $\tau = 0.07$.
  - Nhiệt độ Entropy Gating: $\tau_H = 0.50$.
  - Trọng số loss mức câu: $\lambda_{sent} = 0.08$.
  - Trọng số loss mức token: $\lambda_{tok} = 0.05$.

---

## 4. BẢNG SỐ LIỆU THỰC NGHIỆM ĐỐI CHUẨN ĐẦY ĐỦ (OFFICIAL BENCHMARK RESULTS TABLE)

*Ghi chú: Toàn bộ số liệu dưới đây được đo đạc trực tiếp trên tập kiểm tra gồm 2,295 câu của Tiếng Tày (`tay` $\to$ `vi`), Seed 42, Beam 4.*

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

## 5. CÁC MỐC CHUẨN VÀ RÀNG BUỘC CHO PHƯƠNG PHÁP MỚI (TARGET BENCHMARKS & CONSTRAINTS)

Nhóm đề xuất phương pháp mới cần lưu ý các mốc chỉ số kỹ thuật sau để đảm bảo tính so sánh công bằng:

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
