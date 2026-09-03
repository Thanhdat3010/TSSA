# 🚀 TSSA: Target-Side Semantic Anchoring for Low-Resource Neural Machine Translation

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-orange.svg)](https://pytorch.org/)
[![HuggingFace Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Official PyTorch implementation of **Target-Side Semantic Anchoring (TSSA)**: A self-contained, end-to-end framework for low-resource Neural Machine Translation (NMT) across non-isomorphic ethnic minority language families.

---

## 📑 Danh Mục Tài Liệu Chính Thức (Official Documentation Hub)

Tất cả tài liệu của dự án đã được tinh gọn và chuẩn hóa thành các tài liệu duy nhất sau:

1. **📊 Kết Quả Thực Nghiệm Toàn Diện (Single Source of Truth):**
   * 👉 [`docs/OFFICIAL_EXPERIMENT_RESULTS.md`](docs/OFFICIAL_EXPERIMENT_RESULTS.md) — Bảng so sánh 4 chỉ số (SacreBLEU, chrF++, METEOR, COMET) trên cả 3 ngôn ngữ, phân loại 8 đối thủ baselines (paper links & code), và phân tích Attention Entropy.
2. **🏗️ Kiến Trúc Hệ Thống, Công Thức Toán & Dữ Liệu:**
   * 👉 [`docs/TSSA_SYSTEM_ARCHITECTURE.md`](docs/TSSA_SYSTEM_ARCHITECTURE.md) — Toàn bộ công thức toán học (Online Teacher, Projector, 3 hàm Loss), cơ chế Scheduler và quy trình tiền xử lý 3 bộ dữ liệu chuẩn.
3. **📄 Bản Thảo Báo Cáo Kỹ Thuật (LaTeX Dossier):**
   * 👉 [`docs/tssa_full_paper_dossier.tex`](docs/tssa_full_paper_dossier.tex) — File LaTeX chuẩn quốc tế sẵn sàng biên dịch trên Overleaf.
4. **🌐 Bảng Báo Cáo Trực Quan (Interactive HTML Report):**
   * 👉 [`docs/benchmark_table_report.html`](docs/benchmark_table_report.html) — Giao diện web trực quan để chụp ảnh bảng kết quả.

---

## 🏆 Tóm Tắt Kết Quả Đối Chuẩn (Benchmark Summary - 5 Epochs)

| Ngôn Ngữ Nguồn | Ngữ Hệ | Mô Hình / Phương Pháp | SacreBLEU ↑ | chrF++ ↑ | METEOR ↑ | COMET ↑ |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Ê Đê (`rhade` → `vi`)**<br>*(14,969 train / 1,000 test)* | *Austronesian*<br>*(Nam Đảo)* | Vanilla BARTpho Baseline<br>**TSSA (Ours 🏆)** | 23.41<br>**24.11** *(+0.70)* | 39.33<br>**40.43** *(+1.10)* | 33.91<br>**34.81** *(+0.90)* | -0.3678<br>**-0.3264** *(+0.0414)* |
| **Tày (`tay` → `vi`)**<br>*(20,600 train / 2,295 test)* | *Tai-Kadai*<br>*(Thái-Ka Đai)* | Vanilla BARTpho Baseline<br>**TSSA (Ours 🏆)** | 24.67<br>**25.46** *(+0.79)* | 35.74<br>**36.31** *(+0.57)* | 25.68<br>**26.29** *(+0.61)* | -0.5291<br>**-0.5086** *(+0.0205)* |
| **Ba Na (`bahnaric` → `vi`)**<br>*(51,900 train / 2,001 test)* | *Mon-Khmer*<br>*(Môn-Khơ Me)* | Vanilla BARTpho Baseline<br>**TSSA (Ours 🏆)** | 9.63<br>**9.66** *(+0.03)* | 23.47<br>**23.89** *(+0.42)* | 18.15<br>**18.46** *(+0.31)* | -0.8507<br>**-0.8376** *(+0.0131)* |

---

## 🛠️ Cài Đặt Môi Trường (Installation)

```bash
# 1. Tạo môi trường conda
conda create -n TSSA python=3.10 -y
conda activate TSSA

# 2. Cài đặt PyTorch và các thư viện cần thiết
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

---

## 🚀 Hướng Dẫn Sử Dụng Nhanh (Quick Start)

### 1. Tải và Tiền Xử Lý Dữ Liệu Chuẩn:
```bash
python data/download_and_preprocess.py
```

### 2. Huấn Luyện Mô Hình TSSA (Đề Xuất):
```bash
# Huấn luyện trên tiếng Ê Đê (Rhade)
python train.py --lang rhade --model_type tssa --num_train_epochs 5

# Huấn luyện trên tiếng Tày (Tay)
python train.py --lang tay --model_type tssa --num_train_epochs 5

# Huấn luyện trên tiếng Ba Na (Bahnaric)
python train.py --lang bahnaric --model_type tssa --num_train_epochs 5
```

### 3. Xuất Báo Cáo Kết Quả Toàn Diện (Full 4 Metrics):
```bash
python summary_results.py --comet
```

---

## 📜 Trích Dẫn (Citation)

Nếu bạn sử dụng mã nguồn hoặc kết quả của nghiên cứu này, vui lòng trích dẫn:

```bibtex
@article{tssa2026,
  title={Target-Side Semantic Anchoring and Dynamic Cross-Attention Routing for Low-Resource Neural Machine Translation},
  author={Nguyen, Thanh Dat and Research Team},
  journal={arXiv preprint},
  year={2026}
}
```
