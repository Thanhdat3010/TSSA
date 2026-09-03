# 🏗️ Kiến Trúc Hệ Thống, Công Thức Toán Học & Hướng Dẫn Dữ Liệu (TSSA Architecture & Data Guide)

Tài liệu này tổng hợp toàn bộ **cơ sở lý thuyết, công thức toán học, thiết kế kiến trúc mô hình và quy trình xử lý dữ liệu chuẩn** của phương pháp **Target-Side Semantic Anchoring (TSSA)**.

---

## I. Tổng Quan Kiến Trúc TSSA (System Overview)

TSSA giải quyết triệt để 2 vấn đề lớn trong Dịch máy ít tài nguyên (Low-Resource NMT): **Sụp đổ không gian biểu diễn (Representation Collapse)** và **Chìm đắm chú ý (Attention Sinking)** thông qua 3 module cốt lõi:

```
[Minority Source Sentence X] ---> [Student Encoder f_enc] ---> [Residual Projector Phi] ---> [Mỏ Neo Barycenter L_struct]
                                                                        |                      ^
                                                                        v                      |
[Vietnamese Target Sentence Y] -> [Frozen Teacher E_T]   ---> [Sentence Pooling z^T]   ---> [Ma Trận Dóng Hàng Động A]
                                                                        |
                                                                        v
                                                         [Decoder Dynamic Router L_route] ---> [Sinh Câu Tiếng Việt Dịch]
```

---

## II. Công Thức Toán Học & 3 Hàm Mất Mát (Mathematical Formulation)

### 1. Giáo Viên Trực Tuyến Đóng Băng (Online Frozen Teacher $\mathcal{E}_T$)
* Khóa cứng toàn bộ trọng số của Encoder tiếng Việt:
  $$\mathcal{E}_T = \text{stop\_gradient}(f_{\text{enc}}), \quad h^T = \mathcal{E}_T(\mathcal{Y}) \in \mathbb{R}^{T \times d_{\text{model}}}$$
* Vì dùng chung bộ từ vựng và Tokenizer với Student Decoder, vị trí $t$ của Teacher khớp chính xác $100\%$ với vị trí $t$ của Decoder mà không cần căn chỉnh từ vựng ngoài.

### 2. Ma Trận Dóng Hàng Hậu Nghiệm Động ($A \in \mathbb{R}^{T \times S}$)
* Tính ma trận tương đồng trên GPU trong $0.001\text{s}$:
  $$M_{t, s} = \frac{\tilde{h}_t^T \cdot (\tilde{h}_s^S)^\top}{\tau_{\text{align}}}, \quad \tilde{h} = \frac{h}{\|h\|_2}, \quad \tau_{\text{align}} = 0.1$$
  $$A_{t, s} = \frac{\exp(M_{t, s})}{\sum_{s'=1}^S \exp(M_{t, s'})}, \quad c_s = \min\left(1.0, \sum_{t=1}^T A_{t, s}\right) \cdot \mathbb{I}(x_s \neq \texttt{<pad>})$$

### 3. Bộ Chiếu Phi Tuyến Thặng Dư (Residual Semantic Projector $\Phi_\theta$)
* Chống sụp đổ không gian vector trên tập dữ liệu nhỏ nhờ đường truyền tắt $+ h$:
  $$\Phi_\theta(h) = \text{LayerNorm}(W_2 \cdot \text{GELU}(W_1 h + b_1) + b_2 + h)$$
  *(Trong đó $d_{\text{model}} = 1024, d_{\text{proj}} = 2048$)*.

### 4. Hàm Mất Mát Mỏ Neo Trọng Tâm Mức Từ ($\mathcal{L}_{\text{struct}}$)
* Gom cụm trọng tâm ngữ nghĩa tiếng Việt $\bar{h}_s^T = \sum_{t=1}^T A_{t, s} h_t^T$ và kéo vector từ tiếng dân tộc theo độ tin cậy $c_s \ge 0.2$:
  $$\mathcal{L}_{\text{struct}} = \frac{1}{\sum_s c_s + \epsilon} \sum_{s=1}^S c_s \cdot \text{Smooth}_{L1}\left(\Phi_\theta(h_s^S), \text{sg}(\Phi_\theta(\bar{h}_s^T))\right)$$

### 5. Hàm Mất Mát Tương Phản Toàn Cục Mức Câu ($\mathcal{L}_{\text{prime}}$)
* Ép vector đại diện câu $z^S$ và $z^T$ nằm sát cạnh nhau trong không gian ngữ nghĩa:
  $$\mathcal{L}_{\text{prime}} = -\frac{1}{2B} \sum_{i=1}^B \left[ \log \frac{\exp(z_i^S \cdot z_i^T / \tau)}{\sum_{j=1}^B \exp(z_i^S \cdot z_j^T / \tau)} + \log \frac{\exp(z_i^T \cdot z_i^S / \tau)}{\sum_{j=1}^B \exp(z_i^T \cdot z_j^S / \tau)} \right] \quad (\tau = 0.07)$$

### 6. Cổng Phân Luồng Động ở Decoder ($\mathcal{L}_{\text{route}}$)
* Tạo cổng scalar $g_{\ell h t} \in [0, 1]$ tại từng Head chú ý, được giám sát bởi độ tương đồng Cosine giữa đầu ra Head và trạng thái Teacher $h_t^T$:
  $$r_{\ell h t} = \text{sg}\left(\frac{1 + \cos(R_h o_{\ell h t}, \Phi_\theta(h_t^T))}{2}\right)$$
  $$\mathcal{L}_{\text{route}} = -\frac{1}{|\mathcal{Y}|} \sum_{t=1}^T \sum_{\ell, h} \left[ r_{\ell h t} \log g_{\ell h t} + (1 - r_{\ell h t}) \log (1 - g_{\ell h t}) \right]$$

### 7. Tổng Hàm Mất Mát & Bộ Điều Phối 3 Giai Đoạn (Loss Scheduler)
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MT}} + \lambda_1(t) \mathcal{L}_{\text{struct}} + \lambda_2(t) \mathcal{L}_{\text{prime}} + \lambda_3(t) \mathcal{L}_{\text{route}}$$
* **Giai đoạn 1 (Warmup, Epoch 1--2):** $\lambda_1: 0 \rightarrow 0.10, \lambda_2: 0 \rightarrow 0.05, \lambda_3 = 0.0$ (Ổn định mạng cơ sở).
* **Giai đoạn 2 (Co-adaptation, Epoch 3--4):** $\lambda_1 = 0.10, \lambda_2 = 0.05, \lambda_3: 0 \rightarrow 0.10$ (Kích hoạt cổng Router).
* **Giai đoạn 3 (Fine-tuning, Epoch 5):** Cố định $(\lambda_1, \lambda_2, \lambda_3) = (0.10, 0.05, 0.10)$ và Early Stopping.

---

## III. Hướng Dẫn Dữ Liệu & Quy Trình Tiền Xử Lý Chi Tiết (Comprehensive Data Guide)

Hệ thống hỗ trợ 3 ngữ hệ dân tộc thiểu số với nguồn Hugging Face và các khóa trích xuất (extraction keys) chuẩn:

| Ngôn Ngữ | Ngữ Hệ (*Family*) | Nguồn Hugging Face (*HF Path*) | Khóa Nguồn (*src_key*) | Khóa Đích (*tgt_key*) | Train Mẫu | Test Mẫu |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Ê Đê (`rhade`)** | Nam Đảo (*Austronesian*) | `NIRVLab/rhade-vietnamese-mt` | `ede` / `cdc` | `vi` | 14,969 | 1,000 |
| **Tày (`tay`)** | Thái-Ka Đai (*Tai-Kadai*) | `HeyDunaX/tay-vietnamese-nmt` | `tay` | `viet` / `vietnamese` | 20,600 | 2,295 |
| **Ba Na (`bahnaric`)** | Môn-Khmer (*Mon-Khmer*) | `FiveC/bahnaric_vietnamese` | `bahnaric` | `vietnamese` | 51,900 | 2,001 |

---

### 📦 Chi Tiết Cấu Trúc Cột & JSON Lồng Nhau Trên Hugging Face (HF Columns Schema):

Cả 3 dataset trên Hugging Face đều lưu cặp câu dưới dạng Dictionary lồng nhau bên trong cột `translation` (nested dict):

#### 1. Ba Na – Tiếng Việt (`FiveC/bahnaric_vietnamese`):
* **Cấu trúc JSON trong cột `translation`:**
  ```json
  {
    "bahnaric": "Potao ku'm perm hornet. Anih 'Long...",
    "vietnamese": "Vua lập một. Nơi. Cực. Thánh ở giữa đền. Thờ..."
  }
  ```
* **Splits:** `train` ($\sim 51.9\text{k}$) và `test` ($\sim 2.0\text{k}$).
* **Trích xuất:** `src_key = "bahnaric"`, `tgt_key = "vietnamese"`.

#### 2. Ê Đê – Tiếng Việt (`NIRVLab/rhade-vietnamese-mt`):
* **Cấu trúc JSON trong cột `translation`:**
  ```json
  {
    "ede": "Amai kâo dŏk mă abăn.",
    "vi": "Chị tôi đang lấy chăn."
  }
  ```
* **Splits:** `train` ($14.9\text{k}$) và `test` ($1.0\text{k}$).
* **Trích xuất:** `src_key = ["ede", "cdc"]`, `tgt_key = "vi"`.

#### 3. Tày – Tiếng Việt (`HeyDunaX/tay-vietnamese-nmt`):
* **Cấu trúc JSON trong cột `translation`:**
  ```json
  {
    "tay": "noọng ấc cải",
    "viet": "em ngực bự"
  }
  ```
* **Splits:** `train` ($20.6\text{k}$) và `val` ($2.2\text{k}$).
* **Trích xuất:** `src_key = "tay"`, `tgt_key = ["viet", "vietnamese", "vi"]`. *(Lưu ý: Split `val` trên HuggingFace được đồng nhất lưu thành file `test.csv` cho pipeline kiểm thử)*.

---

### 🛠️ Chi Tiết 4 Bước Tiền Xử Lý Tự Động (`data/download_and_preprocess.py`):

1. **Chuẩn Hóa Unicode Dựng Sẵn (Unicode NFC):**
   * Sử dụng `unicodedata.normalize('NFC', text)` cho toàn bộ câu tiếng nguồn và tiếng đích để đồng nhất các ký tự có dấu thanh, phụ tố và nguyên âm biến âm của chữ Ê Đê, Tày, Ba Na.
2. **Làm Sạch Ký Tự Rác & Khoảng Trắng Thừa (Regex Cleaning):**
   * Xóa ký tự xuống dòng rác: `re.sub(r'\\n|\|\\|[\n\r\t]', ' ', text)`.
   * Chuẩn hóa khoảng trắng kép: `re.sub(r'\s+', ' ', text).strip()`.
   * Lọc bỏ hoàn toàn các cặp câu rỗng hoặc bị gán nhãn `None`.
3. **Bảo Vệ Tính Toàn Vẹn & Khử Rò Rỉ Tuyệt Đối (Strict Leak-Free Deduplication):**
   * Giải nén dictionary `translation` thành 2 cột phẳng: `src_text` (tiếng dân tộc) và `tgt_text` (tiếng Việt).
   * Loại bỏ trùng lặp nội bộ: `df.drop_duplicates()`.
   * Khử rò rỉ tập kiểm thử: `train_df = train_df[~train_df["src_text"].isin(test_df["src_text"])]`. Đảm bảo không có bất kỳ câu nào trong tập kiểm thử xuất hiện trong tập huấn luyện.
4. **Định Dạng Lưu Trữ Chuẩn & Cắt Độ Dài (Truncation):**
   * Dữ liệu xuất ra file `data_processed/<lang>/train.csv` và `data_processed/<lang>/test.csv` gồm 2 cột chuẩn: `src_text` và `tgt_text` (mã hóa `utf-8`).
   * Khi đưa vào DataLoader, áp dụng cắt độ dài chuẩn `max_source_length = 256` và `max_target_length = 256` bằng tokenizer `vinai/bartpho-syllable`.


---

## IV. Cấu Trúc Mã Nguồn Dự Án (Codebase Architecture)

```
TSSA/
├── models/
│   ├── teacher_wrapper.py        # Module Online Frozen Teacher (sg)
│   ├── semantic_projector.py     # Module Residual Projector Phi (1024 -> 2048 -> 1024)
│   ├── head_router.py            # Module Dynamic Cross-Attention Routing Gate
│   └── tssa_seq2seq.py           # Backbone tích hợp toàn diện TSSA Seq2Seq
├── losses/
│   ├── struct_loss.py            # Hàm Loss mỏ neo mức từ (Barycenter + Smooth L1)
│   ├── prime_loss.py             # Hàm Loss tương phản mức câu (InfoNCE tau=0.07)
│   ├── route_loss.py             # Hàm Loss cổng Decoder (Soft BCE)
│   └── unified_criterion.py      # Bộ tổng hợp đa hàm mất mát Multi-task
├── training/
│   ├── loss_scheduler.py         # Bộ điều phối trọng số Loss 3 giai đoạn
│   └── trainer.py                # Seq2Seq Trainer tùy biến cho TSSA & Baselines
├── summary_results.py            # Công cụ trích xuất báo cáo full 4 metrics & cache JSON
└── train.py                      # File thực thi huấn luyện trung tâm
```
