# Hướng Dẫn Cấu Trúc Thư Mục Code & Nguồn Dữ Liệu Thực Nghiệm (TSSA)

Tài liệu này đặc tả chi tiết **nguồn dữ liệu thực tế (Datasets)** trích xuất từ file `TSSA.pdf` và cấu trúc cột Hugging Face, **mã nguồn tải & tiền xử lý chuẩn hóa**, kèm theo **thiết kế cây thư mục code (Project Architecture)** theo chuẩn kỹ thuật phần mềm để bạn có thể bắt tay vào lập trình Python ngay lập tức.

---

## I. Nguồn Dữ Liệu Thực Nghiệm & Cấu Trúc Cột (Datasets Schema)

Theo hình ảnh thực tế trên **Hugging Face Hub** và **Mục 4.1 trong TSSA.pdf**, cả 3 tập dữ liệu đều sử dụng định dạng chuẩn `translation` dạng dictionary (nested dict), nhưng có sự khác biệt nhỏ về tên khóa (keys):

```
+----------------------------------------------------------------------------------------------------------------------+
| Ngôn ngữ           | Hugging Face Repo Path                         | Splits trên Hugging Face | Cấu trúc Key trong 'translation'      |
+--------------------+------------------------------------------------+--------------------------+---------------------------------------+
| 1. Ba Na (Bahnar)  | FiveC/bahnaric_vietnamese                      | train (51.9k), test      | {'bahnaric': '...', 'vietnamese': '...'}|
| 2. Ê Đê (Rhade)    | NIRVLab/rhade-vietnamese-mt                    | train (15.1k), test      | {'ede': '...', 'vi': '...'}           |
| 3. Tày (Tay)       | HeyDunaX/tay-vietnamese-nmt                    | train (20.6k), val       | {'tay': '...', 'viet': '...'}         |
+----------------------------------------------------------------------------------------------------------------------+
```

### 1. Chi tiết cấu trúc từng tập dữ liệu:

#### 🟢 Tập 1: Ba Na – Tiếng Việt (`FiveC/bahnaric_vietnamese`)
* **Splits trên Hugging Face:** Gồm 2 splits chuẩn: **`train`** ($\sim 51.900$ dòng) và **`test`**.
* **Cấu trúc JSON/Dict trong cột `translation`:**
  ```json
  {
    "bahnaric": "Potao ku'm perm hornet. Anih 'Long...",
    "vietnamese": "Vua lập một. Nơi. Cực. Thánh ở giữa đền. Thờ..."
  }
  ```
* **Source Key:** `"bahnaric"` | **Target Key:** `"vietnamese"`.
* **Đầu ra:** Lưu trực tiếp `train` $\rightarrow$ `train.csv` và `test` $\rightarrow$ `test.csv`.

#### 🔵 Tập 2: Ê Đê – Tiếng Việt (`NIRVLab/rhade-vietnamese-mt`)
* **Splits trên Hugging Face:** Gồm 2 splits chuẩn: **`train`** ($15.100$ dòng) và **`test`** (tổng $\sim 17.092$ dòng).
* **Cấu trúc JSON/Dict trong cột `translation`:**
  ```json
  {
    "ede": "Amai kâo dŏk mă abăn.",
    "vi": "Chị tôi đang lấy chăn."
  }
  ```
* **Source Key:** `"ede"` (hoặc `"cdc"`) | **Target Key:** `"vi"`.
* **Đầu ra:** Lưu trực tiếp `train` $\rightarrow$ `train.csv` và `test` $\rightarrow$ `test.csv`.

#### 🟡 Tập 3: Tày – Tiếng Việt (`HeyDunaX/tay-vietnamese-nmt`)
* **Splits trên Hugging Face:** Gồm 2 splits: **`train`** ($20.600$ dòng) và **`val`**.
* **Cấu trúc JSON/Dict trong cột `translation`:**
  ```json
  {
    "tay": "noọng ấc cải",
    "viet": "em ngực bự"
  }
  ```
* **Source Key:** `"tay"` | **Target Key:** `"viet"` *(Lưu ý: Khóa đích là `"viet"`)*.
* **Đầu ra:** Lưu `train` $\rightarrow$ `train.csv` và `val` $\rightarrow$ `test.csv` (đồng nhất thành file `test.csv` cho toàn bộ pipeline).

---

## II. Tiêu Chuẩn Tiền Xử Lý Dữ Liệu (Sử Dụng Trực Tiếp Splits Có Sẵn)

1. **Sử dụng trực tiếp Splits chuẩn từ Hugging Face:**
   * Cả 3 datasets trên Hugging Face đều **đã được chia sẵn các splits chính thức**.
   * Với Ba Na & Ê Đê: Trích xuất `train` $\rightarrow$ `train.csv` và `test` $\rightarrow$ `test.csv`.
   * Với Tày: Trích xuất `train` $\rightarrow$ `train.csv` và `val` $\rightarrow$ `test.csv` (đồng nhất tên file `test.csv` cho toàn bộ dự án).
2. **Chuẩn hóa cột đầu ra:** Toàn bộ dữ liệu sau khi tải về sẽ được giải nén từ dictionary `translation` thành 2 cột phẳng đồng nhất: `src_text` (Tiếng hiếm) và `tgt_text` (Tiếng Việt).
3. **Lưu trữ chuẩn hóa:** Lưu trực tiếp thành 2 file sạch:
   * `data_processed/<lang>/train.csv`
   * `data_processed/<lang>/test.csv`
4. **Chuẩn hóa Unicode:** Áp dụng `unicodedata.normalize('NFC', text)` cho cả tiếng nguồn và tiếng đích để đồng nhất bảng mã tiếng Việt và các ký tự có dấu của tiếng Ba Na, Ê Đê, Tày.

---

## III. Thiết Kế Cấu Trúc Thư Mục Dự Án Code (Project Layout)

```text
d:\Code\Mapping\
├── docs/                                  # Tài liệu thiết kế và nghiên cứu
│   ├── TSSA.pdf                           # File bài báo gốc
│   ├── TSSA_Introduction.md               # Bản thảo Introduction
│   ├── TSSA_Methodology.md                # Bản thảo chi tiết Phương pháp
│   ├── TSSA_Experiment_Design_and_Code_Blueprint.md # Kế hoạch thí nghiệm & Ablation
│   └── TSSA_Code_Architecture_and_Data_Guide.md     # Tài liệu hướng dẫn này
│
├── configs/                               # File cấu hình Hyperparameters (.yaml)
│   ├── base_config.yaml                   # Cấu hình chung (Learning rate, batch size, epochs)
│   ├── data_configs/                      # Cấu hình đường dẫn dataset từng ngôn ngữ (train/test)
│   │   ├── bahnaric.yaml
│   │   ├── tay.yaml
│   │   └── rhade.yaml
│   └── ablation_configs/                  # Cấu hình cho các thí nghiệm Ablation
│       ├── component_ablation.yaml
│       ├── teacher_mode_ablation.yaml
│       └── causal_pruning.yaml
│
├── data/                                  # Module xử lý dữ liệu & Aligner
│   ├── __init__.py
│   ├── download_and_preprocess.py         # Script tự động tải, giải nén dict, xuất train.csv & test.csv
│   ├── word_aligner.py                    # Chạy SimAlign / AWESOME-align xuất ma trận A
│   ├── teacher_caching.py                 # Chạy trước forward pass BARTpho để cache vector đích
│   └── dataloader.py                      # PyTorch Dataset & DataLoader
│
├── models/                                # Module Kiến trúc Mô hình (PyTorch)
│   ├── __init__.py
│   ├── teacher_wrapper.py                 # Wrapper mô hình BARTpho đóng băng (Frozen Teacher)
│   ├── student_backbone.py                # Transformer scratch / BARTpho adapted cho tiếng hiếm
│   ├── head_router.py                     # Cổng MLP Cross-Attention Head Router
│   └── tssa_seq2seq.py                    # Mô hình TSSA hoàn chỉnh tích hợp Router
│
├── losses/                                # Module các Hàm Mất Mát Toán Học
│   ├── __init__.py
│   ├── struct_loss.py                     # L_struct: Confidence-Weighted Barycentric Anchoring
│   ├── prime_loss.py                      # L_prime: In-Batch InfoNCE Semantic Priming
│   ├── route_loss.py                      # L_route: Soft BCE Supervision cho Head Router
│   └── unified_criterion.py               # Tổng hợp: L_total = L_MT + l1*L_struct + l2*L_prime + l3*L_route
│
├── training/                              # Module Huấn Luyện & Scheduler
│   ├── __init__.py
│   ├── loss_scheduler.py                  # Điều phối tăng dần lambda_1, lambda_2, lambda_3
│   ├── trainer.py                         # Vòng lặp Train / Validate / Checkpoint
│   └── optimizer_utils.py                 # Cấu hình AdamW, Linear Warmup, Gradient Clipping
│
├── evaluation/                            # Module Đánh Giá & Thẩm Định Nhân Quả
│   ├── __init__.py
│   ├── evaluator.py                       # Chấm điểm sacreBLEU, chrF++, COMET trên test.csv
│   ├── causal_head_pruning.py             # Thí nghiệm cắt tỉa Top-K vs Random-K Heads
│   ├── robustness_noise.py                # Sinh nhiễu (xóa dấu, sai chính tả, đảo từ) trên test.csv
│   └── alignment_metrics.py               # Đo lường Alignment Precision, Recall, F1
│
├── scripts/                               # Scripts chạy tự động hóa (Bash / PowerShell)
│   ├── 01_prepare_all_data.ps1            # Tải dataset + sinh SimAlign matrix + cache Teacher
│   ├── 02_run_main_benchmark.ps1          # Chạy toàn bộ thí nghiệm chính
│   └── 03_run_ablations.ps1               # Chạy toàn bộ các thí nghiệm Ablation & Causal
│
├── data_processed/                        # Thư mục lưu dữ liệu sạch sau khi tải
│   ├── bahnaric/                          # train.csv, test.csv
│   ├── rhade/                             # train.csv, test.csv
│   └── tay/                               # train.csv, test.csv
│
├── checkpoints/                           # Thư mục lưu trọng số model (.pt / .safetensors)
├── outputs/                               # Thư mục lưu kết quả dịch, log tensorboard & biểu đồ
├── requirements.txt                       # Danh sách các thư viện Python cần thiết
├── train.py                               # File chạy chính cho Huấn luyện
└── evaluate.py                            # File chạy chính cho Đánh giá / Test
```

---

---

## V. Tích Hợp Kỹ Thuật Từ Code Mẫu `bartbana_final.py`

File code thực nghiệm của lab bạn (`bartbana_final.py`) đã thiết lập chuẩn mực chạy thực tế trên GPU A100. Dưới đây là cách kế thừa và nâng cấp vào hệ thống TSSA:

### 1. Các thành phần kế thừa nguyên bản từ `bartbana_final.py`:
* **Độ dài Token tối đa:** `MAX_SOURCE_LENGTH = 256`, `MAX_TARGET_LENGTH = 256` *(Mở rộng lên 256 token để bao phủ đầy đủ các câu phức và câu dài)*.
* **Hàm làm sạch văn bản:** Chuẩn hóa Regex `\n`, `\r`, `\t` và khoảng trắng thừa.
* **Tokenization 2 chiều:** Sử dụng `AutoTokenizer.from_pretrained(MODEL_CKPT)` kèm `tokenizer.as_target_tokenizer()` cho phần nhãn đích.
* **Collator & Trainer Arguments:**
  * `DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)`
  * `learning_rate = 2e-5`, `weight_decay = 0.01`, `fp16 = True` (tối ưu tốc độ trên GPU).
  * `EarlyStoppingCallback(early_stopping_patience = 3)`
  * `metric_for_best_model = "sacrebleu"` (kèm `chrF++` và `COMET`).

---

### 2. Nâng cấp: `TSSASeq2SeqTrainer` (Kế thừa trực tiếp `Seq2SeqTrainer`)

Thay vì viết lại vòng lặp PyTorch thủ công, TSSA kế thừa trực tiếp từ `Seq2SeqTrainer` của Hugging Face và override hàm `compute_loss()`. Điều này giúp tận dụng 100% tính năng cao cấp (FP16, Gradient Accumulation, Checkpoint, Multi-GPU) mà vẫn tính toán chính xác 3 hàm mất mát của TSSA:

```python
import torch
from transformers import Seq2SeqTrainer

class TSSASeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, *args, struct_loss_fn=None, prime_loss_fn=None, route_loss_fn=None, loss_scheduler=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.struct_loss_fn = struct_loss_fn
        self.prime_loss_fn = prime_loss_fn
        self.route_loss_fn = route_loss_fn
        self.loss_scheduler = loss_scheduler

    def compute_loss(self, model, inputs, return_outputs=False):
        # 1. Forward Student Model để lấy Cross-Entropy Loss (L_MT) và Hidden States
        outputs = model(**inputs, output_hidden_states=True, output_attentions=True)
        loss_mt = outputs.loss
        
        # Lấy trọng số loss tại bước huấn luyện hiện tại từ LossScheduler
        current_step = self.state.global_step
        l1, l2, l3 = self.loss_scheduler.get_lambdas(current_step) if self.loss_scheduler else (0.5, 0.2, 0.1)
        
        total_loss = loss_mt
        
        # 2. Tính L_struct (Token Barycenter Anchoring) nếu được bật
        if self.struct_loss_fn is not None and l1 > 0:
            student_enc_states = outputs.encoder_hidden_states[-1] # [B, S, D]
            teacher_enc_states = inputs.get("teacher_enc_states")   # [B, T, D] từ frozen teacher
            align_matrix = inputs.get("align_matrix")               # [B, S, T] từ SimAlign
            
            l_struct = self.struct_loss_fn(student_enc_states, teacher_enc_states, align_matrix)
            total_loss = total_loss + l1 * l_struct
            
        # 3. Tính L_prime (In-batch Sentence InfoNCE) nếu được bật
        if self.prime_loss_fn is not None and l2 > 0:
            src_sent_vec = outputs.encoder_hidden_states[-1][:, 0, :] # [B, D] (Token đầu hoặc mean-pool)
            tgt_sent_vec = inputs.get("teacher_sent_vec")             # [B, D]
            
            l_prime = self.prime_loss_fn(src_sent_vec, tgt_sent_vec)
            total_loss = total_loss + l2 * l_prime
            
        # 4. Tính L_route (Decoder Head-wise Router Loss) nếu được bật
        if self.route_loss_fn is not None and l3 > 0 and hasattr(model, "router_gates"):
            router_logits = model.router_gates                        # [B, Layers, Heads, T]
            teacher_reliability = inputs.get("teacher_reliability")   # [B, Layers, Heads, T]
            
            l_route = self.route_loss_fn(router_logits, teacher_reliability)
            total_loss = total_loss + l3 * l_route
            
        return (total_loss, outputs) if return_outputs else total_loss
```

---

### 3. Thiết Kế Bộ Tham Số Dòng Lệnh Hoàn Chỉnh (`train.py`)

Cấu trúc dòng lệnh `argparse` được thiết kế tương thích hoàn toàn với các tham số chuẩn ban đầu:

```python
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="TSSA & Baselines Unified Training Suite")
    
    # 1. Tham số dữ liệu & Ngôn ngữ
    parser.add_argument("--lang", type=str, default="bahnaric", choices=["rhade", "tay", "bahnaric"],
                        help="Ngôn ngữ dân tộc thiểu số cần huấn luyện")
    parser.add_argument("--data_dir", type=str, default="data_processed", help="Thư mục chứa train.csv và test.csv")
    parser.add_argument("--max_source_length", type=int, default=256, help="Độ dài tối đa câu nguồn")
    parser.add_argument("--max_target_length", type=int, default=256, help="Độ dài tối đa câu đích")
    
    # 2. Tham số Mô hình Backbone & Checkpoint
    parser.add_argument("--model_ckpt", type=str, default="vinai/bartpho-syllable", 
                        help="Backbone Pretrained Model (ví dụ: vinai/bartpho-syllable hoặc IAmSkyDra/BARTBana_v5)")
    parser.add_argument("--teacher_ckpt", type=str, default="vinai/bartpho-syllable", help="Teacher Backbone (Frozen)")
    parser.add_argument("--model_type", type=str, default="tssa", 
                        choices=["tssa", "bartpho_vanilla", "joint_align", "guided_attn", "awesome_align", "cl_lsa"],
                        help="Loại mô hình / phương pháp đối thủ cần chạy")
    
    # 3. Tham số Hàm Mất Mát TSSA (Ablation Flags)
    parser.add_argument("--use_struct", action="store_true", default=True, help="Bật L_struct (Token Barycenter)")
    parser.add_argument("--use_prime", action="store_true", default=True, help="Bật L_prime (Sentence InfoNCE)")
    parser.add_argument("--use_route", action="store_true", default=True, help="Bật L_route (Decoder Head-wise Router)")
    parser.add_argument("--teacher_mode", type=str, default="frozen", choices=["frozen", "trainable", "ema"])
    parser.add_argument("--lambda_struct", type=float, default=0.5)
    parser.add_argument("--lambda_prime", type=float, default=0.2)
    parser.add_argument("--lambda_route", type=float, default=0.1)
    
    # 4. Tham số Huấn luyện & Tối ưu hóa
    parser.add_argument("--batch_size", type=int, default=16, help="Kích thước batch trên mỗi GPU")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate chuẩn (theo bartbana_final)")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--fp16", action="store_true", default=True, help="Bật Mixed Precision FP16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    
    return parser.parse_args()
```

---

## VI. Môi Trường Huấn Luyện (Conda Environment: `TSSA`) & Thư Viện (`requirements.txt`)

### 1. Khởi tạo môi trường Conda:
* **Tên môi trường Conda (Env Name):** **`TSSA`**
* **Phiên bản Python chuẩn:** **`Python 3.10`**

```bash
# Lệnh tạo và kích hoạt môi trường:
conda create -n TSSA python=3.10 -y
conda activate TSSA

# Cài đặt PyTorch với CUDA hỗ trợ:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Cài đặt toàn bộ thư viện dự án:
pip install -r requirements.txt
```

### 2. Danh sách thư viện (`requirements.txt`):

```text
torch>=2.1.0
transformers>=4.38.0
datasets>=2.18.0
accelerate>=0.28.0
evaluate>=0.4.1
sacrebleu>=2.4.0
unbabel-comet>=2.2.0
sentencepiece>=0.2.0
simalign>=0.3
pyyaml>=6.0
pandas>=2.1.0
tqdm>=4.66.0
tensorboard>=2.15.0
```
