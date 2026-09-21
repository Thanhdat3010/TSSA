# 📋 QUY CHUẨN CẤU HÌNH THỰC NGHIỆM ĐỐI CHUẨN CÔNG BẰNG (FAIR BENCHMARK SETTINGS)

> **Mục tiêu:** Thiết lập một bộ thiết lập thực nghiệm duy nhất, bất biến và hoàn toàn đối xứng (Symmetric & Strictly Fair) để chạy lại **Vanilla Baseline**, **các phương pháp cạnh tranh quốc tế (Baselines)** và **TSSA-Pro** với cùng 1 cấu hình và môi trường trên máy chủ GPU.

---

## 1. NGUYÊN TẮC THIẾT KẾ CÔNG BẰNG (FAIRNESS PRINCIPLES)

1. **Một script huấn luyện duy nhất (`train_fair_benchmark.py`):**
   * Toàn bộ các phương pháp đều sử dụng cùng một kiến trúc cốt lõi, cùng một bộ khởi tạo `Seq2SeqTrainer`, cùng một hàm tối ưu và cùng một hàm suy luận đánh giá.
   * Loại bỏ hoàn toàn sự sai lệch do khác biệt script hoặc cấu hình ngầm định.

2. **Dữ liệu huấn luyện và đánh giá cố định:**
   * Giữ nguyên cấu trúc dữ liệu hiện tại trong `data_processed/<lang>/`:
     - **`train.csv`:** Dùng cho huấn luyện mô hình.
     - **`test.csv`:** Dùng cho đánh giá định kỳ sau mỗi epoch (`eval_dataset`) để kích hoạt dừng sớm (Early Stopping) và chọn checkpoint tốt nhất (`load_best_model_at_end=True`), đồng thời xuất điểm số cuối cùng.

3. **Kiểm soát tính ngẫu nhiên (RNG & Seed Independence):**
   * Thiết lập đồng bộ: `transformers.set_seed(seed)`, `torch.manual_seed(seed)`, `torch.cuda.manual_seed_all(seed)`, `np.random.seed(seed)`.
   * Truyền đồng thời `seed` và `data_seed` vào `Seq2SeqTrainingArguments` để đảm bảo xáo trộn dữ liệu (shuffling) độc lập qua các seed 42, 43, 44.

---

## 2. BẢNG THÔNG SỐ HUẤN LUYỆN CHUẨN (UNIFIED HYPERPARAMETERS)

| Tham Số (*Hyperparameter*) | Giá Trị Cố Định (BARTpho) | Giá Trị Cố Định (ViT5) | Ghi Chú Kỹ Thuật |
| :--- | :---: | :---: | :--- |
| **Pretrained Backbone** | `vinai/bartpho-syllable` | `VietAI/vit5-base` | Mô hình gốc từ Hugging Face Hub |
| **Số Epoch Tối Đa** | **5** | **5** | Có Early Stopping bảo vệ |
| **Early Stopping** | `patience = 3` | `patience = 3` | Giám sát `sacrebleu` trên `test.csv` |
| **Batch Size (Per Device)** | **16** | **16** | Khớp VRAM A100 / RTX 4090 |
| **Tốc Độ Học (*Learning Rate*)** | **`2e-5`** | **`1e-4`** | Chuẩn hội tụ ổn định của từng backbone |
| **Bộ Tối Ưu (*Optimizer*)** | **AdamW** | **AdamW** | $\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-8}$ |
| **Weight Decay** | **0.01** | **0.01** | Chống overfitting |
| **Warmup Ratio / Steps** | **10% (hoặc 500 steps)** | **10% (hoặc 500 steps)** | Linear Warmup |
| **Độ Chính Xác Số Học** | **FP16** | **BF16** | ViT5 dùng BF16 để chống tràn số T5 |
| **Max Source Length** | **256** tokens | **256** tokens | Bao phủ 99.8% độ dài câu nguồn |
| **Max Target Length** | **256** tokens | **256** tokens | Bao phủ 99.8% độ dài câu đích |
| **Metric for Best Model** | `sacrebleu` | `sacrebleu` | Đánh giá sau mỗi epoch trên tập test |

---

## 3. CẤU HÌNH GIẢI MÃ ĐÁNH GIÁ (DECODING & INFERENCE PROTOCOL)

Toàn bộ các mô hình sau khi chọn best checkpoint sẽ được giải mã bằng một hàm chung:

```python
generated_ids = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_length=256,
    num_beams=4,
    length_penalty=1.0,
    early_stopping=True
)
```

### Bộ 4 Chỉ Số Đánh Giá Chuẩn Quốc Tế:
1. **SacreBLEU:** `sacrebleu.corpus_bleu(predictions, [references], smooth_method="exp")`
2. **chrF++:** `sacrebleu.corpus_chrf(predictions, [references], word_order=2)`
3. **METEOR:** NLTK WordNet Vietnamese/Universal translation aligner score.
4. **COMET:** Model `Unbabel/wmt22-comet-da` (chạy trên GPU).

---

## 4. DANH SÁCH CÁC PHƯƠNG PHÁP ĐƯỢC CHẠY ĐỐI CHUẨN

Tất cả các phương pháp dưới đây sẽ được gọi thông qua cùng một script với cờ `--model_type`:

1. **`vanilla` (Mô hình cơ sở chuẩn):** Huấn luyện Seq2Seq chuẩn, không có loss phụ.
2. **`awesome_align` (EACL 2021):** Giám sát ma trận căn chỉnh từ tố hai chiều thông qua self-training.
3. **`cl_lsa` (NAACL 2021):** Căn chỉnh biểu diễn câu thông qua hàm mất mát tương phản Cross-Lingual InfoNCE.
4. **`align_to_distill` (COLING 2024):** Chưng cất phân phối Cross-Attention từ Teacher.
5. **`shift_aet` (EMNLP 2020):** Giám sát trạng thái tịnh tiến tự hồi quy.
6. **`tssa_pro` (Đề xuất):** Mỏ neo biểu diễn ngữ nghĩa trọng tâm kết hợp cổng động và suy giảm theo phân mảnh từ tố $\kappa$.

---

## 5. CẤU TRÚC THƯ MỤC LƯU TRỮ ĐỘC LẬP

Để tránh tuyệt đối việc ghi đè hoặc nhầm lẫn với các lần chạy trước:
* Thư mục lưu checkpoint: `checkpoints/fair_benchmark/<model_type>_<lang>_seed<seed>/`
* Thư mục báo cáo tổng hợp: `checkpoints/fair_benchmark/FAIR_BENCHMARK_REPORT.md`
* Mỗi thư mục checkpoint chứa đầy đủ:
  - `pytorch_model.bin` / `model.safetensors` (Best Model)
  - `tokenizer_config.json` và vocabulary
  - `eval_metrics.json` (Ghi nhận kết quả 4 chỉ số trên Test)
  - `test_predictions.csv` (Bản dịch thực tế từng câu để chạy Paired Bootstrap)

---

## 6. GIAO THỨC SMOKE TEST (KIỂM TRA AN TOÀN TRƯỚC KHI CHẠY THẬT)

Trước khi chạy toàn bộ tập dữ liệu trên GPU, script smoke test `scripts/smoke_test_fair_benchmark.py` phải xác nhận:
* [x] Nạp dữ liệu `train.csv` và `test.csv` và tokenizer không lỗi.
* [x] Chạy 20 gradient steps cho mỗi phương pháp không bị NaN/Inf loss.
* [x] Chạy Vanilla với seed 42 và seed 43 cho ra loss khác nhau (xác nhận Seed độc lập hoạt động).
* [x] Lưu và nạp lại checkpoint thành công.
* [x] Sinh bản dịch bằng Beam Search 4 và tính điểm SacreBLEU, chrF++ thành công.
