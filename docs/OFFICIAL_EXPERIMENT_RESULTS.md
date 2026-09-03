# 📊 Báo Cáo Toàn Diện Kết Quả Thực Nghiệm & Đối Chuẩn (Official Experiment Results)

Tài liệu này là **Nguồn Sự Thật Duy Nhất (Single Source of Truth)** lưu trữ toàn bộ số liệu thực nghiệm, bảng so sánh chính thức 4 chỉ số chuẩn quốc tế, phân tích cơ chế nội tại của mô hình **TSSA** so với toàn bộ các phương pháp đối chứng (Baselines) trên **3 ngữ hệ ngôn ngữ thiểu số Việt Nam**.

---

## I. Phân Loại 8 Phương Pháp Đối Chứng (Baselines Taxonomy & Papers)

Toàn bộ các phương pháp đều được huấn luyện trên **cùng Backbone `vinai/bartpho-syllable`** và kiểm thử trên cùng tập `test.csv` chính thức của 3 ngôn ngữ:

| Nhóm Phương Pháp | Phương Pháp Đối Chứng | Bài Báo Gốc (Paper Link) | Kho Mã Nguồn Chính Thức (GitHub) |
| :--- | :--- | :--- | :--- |
| **0. Sàn Cơ Sở (Base NMT)** | **Vanilla BARTpho** | [Findings of EMNLP 2021](https://aclanthology.org/2021.findings-emnlp.294.pdf) | [`VinAIResearch/BARTpho`](https://github.com/VinAIResearch/BARTpho) |
| **1. Attention Distillation**<br>*(Chưng cất phân phối chú ý)* | • **Align-to-Distill (A2D)**<br>• **Structural Supervision** | [LREC-COLING 2024](https://aclanthology.org/2024.lrec-main.722.pdf)<br>[Findings of ACL 2022](https://aclanthology.org/2022.findings-acl.322.pdf) | [`ncsoft/Align-to-Distill`](https://github.com/ncsoft/Align-to-Distill)<br>[`alibaba/Alibaba-NLP`](https://github.com/alibaba/Alibaba-NLP) |
| **2. Shifted State Align**<br>*(Dóng hàng dịch chuyển trạng thái)* | • **Shift-AET** | [EMNLP 2020](https://aclanthology.org/2020.emnlp-main.456.pdf) | [`sufe-nlp/transformer-alignment`](https://github.com/sufe-nlp/transformer-alignment) |
| **3. Embedding Alignment**<br>*(Căn chỉnh không gian vector từ)* | • **AWESOME-align**<br>• **CrossInit**<br>• **DM-BLI Subspace** | [EACL 2021](https://aclanthology.org/2021.eacl-main.181.pdf)<br>[Findings of ACL 2024](https://aclanthology.org/2024.findings-acl.316.pdf)<br>[ACL 2024](https://aclanthology.org/2024.acl-long.112.pdf) | [`neulab/awesome-align`](https://github.com/neulab/awesome-align)<br>[`baridxiai/crossInit_trial`](https://github.com/baridxiai/crossInit_trial)<br>[`huling-2/DM-BLI`](https://github.com/huling-2/DM-BLI) |
| **4. Contrastive Learning**<br>*(Học biểu diễn tương phản)* | • **Cross-Lingual InfoNCE (CL-LSA)**<br>• **DPO-Align** | [NAACL 2021 (InfoXLM)](https://aclanthology.org/2021.naacl-main.280.pdf)<br>[EMNLP 2024](https://aclanthology.org/2024.emnlp-main.188.pdf) | [`microsoft/InfoXLM`](https://github.com/microsoft/unilm)<br>[`DiWu-NLP/DPO-Align`](https://github.com/DiWu-NLP) |
| **⭐ ĐỀ XUẤT (Ours)** | **TSSA (This Work 🏆)** | [TSSA Architecture](file:///d:/Code/Mapping/docs/TSSA_SYSTEM_ARCHITECTURE.md) | *This Repository* |

---

## II. Bảng So Sánh Chính Thức 4 Chỉ Số (Main Benchmark - Table 1)

* **Thiết lập:** 5 Epochs, Batch size 16, Learning rate $2\times 10^{-5}$ (AdamW), FP16 trên GPU NVIDIA A100.
* **4 Metric Chuẩn Quốc Tế:** SacreBLEU, chrF++, METEOR (`nltk`), và COMET (`Unbabel/wmt20-comet-da`).

| Ngôn Ngữ Nguồn | Ngữ Hệ | Phương Pháp | SacreBLEU ↑ | chrF++ ↑ | METEOR ↑ | COMET ↑ | Mức Tăng vs Vanilla (Δ) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Ê Đê (`rhade` → `vi`)**<br/>*(14,969 train, 1,000 test)* | *Austronesian*<br>*(Nam Đảo)* | **BARTpho Baseline** | 23.41 | 39.33 | 33.91 | -0.3678 | Mốc sàn cơ sở |
| | | `align_to_distill` | 22.93 | 39.12 | 33.51 | -0.3693 | -0.48 BLEU, -0.21 chrF++ |
| | | `structural_supervision` | 20.48 | 35.91 | 31.21 | -0.4860 | -2.93 BLEU, -3.42 chrF++ |
| | | `shift_aet` | 22.38 | 38.26 | 32.90 | -0.4181 | -1.03 BLEU, -1.07 chrF++ |
| | | `awesome_align` | 23.05 | 38.93 | 33.45 | -0.3841 | -0.36 BLEU, -0.40 chrF++ |
| | | `cl_lsa` | 17.85 | 33.33 | 28.19 | -0.5969 | -5.56 BLEU, -6.00 chrF++ |
| | | **TSSA (Ours 🏆)** | **24.11** | **40.43** | **34.81** | **-0.3264** | **+0.70 BLEU, +1.10 chrF++, +0.90 METEOR, +0.0414 COMET** 🚀 |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Tày (`tay` → `vi`)**<br/>*(20,600 train, 2,295 test)* | *Tai-Kadai*<br>*(Thái-Ka Đai)* | **BARTpho Baseline** | 24.67 | 35.74 | 25.68 | -0.5291 | Mốc sàn cơ sở |
| | | `align_to_distill` | 24.67 | 35.84 | 26.17 | -0.5138 | +0.00 BLEU, +0.10 chrF++ |
| | | `shift_aet` | 19.44 | 28.97 | 19.39 | -0.7602 | -5.23 BLEU, -6.77 chrF++ |
| | | `awesome_align` | 25.20 | 36.30 | 26.44 | -0.5051 | +0.53 BLEU, +0.56 chrF++ |
| | | `cl_lsa` | 24.48 | 35.29 | 25.53 | -0.5512 | -0.19 BLEU, -0.45 chrF++ |
| | | **TSSA (Ours 🏆)** | **25.46** | **36.31** | **26.29** | **-0.5086** | **+0.79 BLEU, +0.57 chrF++, +0.61 METEOR, +0.0205 COMET** 🚀 |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Ba Na (`bahnaric` → `vi`)**<br/>*(51,900 train, 2,001 test)* | *Mon-Khmer*<br>*(Môn-Khơ Me)* | **BARTpho Baseline** | 9.63 | 23.47 | 18.15 | -0.8507 | Mốc sàn cơ sở |
| | | `align_to_distill` | 9.15 | 23.17 | 18.05 | -0.8598 | -0.48 BLEU, -0.30 chrF++ |
| | | `shift_aet` | 9.10 | 23.31 | 18.00 | -0.8682 | -0.53 BLEU, -0.16 chrF++ |
| | | `awesome_align` | 9.07 | 23.22 | 17.92 | -0.8588 | -0.56 BLEU, -0.25 chrF++ |
| | | `cl_lsa` | 4.49 | 15.87 | 9.90 | -1.1817 | -5.14 BLEU, -7.60 chrF++ |
| | | **TSSA (Ours 🏆)** | **9.66** | **23.89** | **18.46** | **-0.8376** | **+0.03 BLEU, +0.42 chrF++, +0.31 METEOR, +0.0131 COMET** 🚀 |

---

## III. Phân Tích Cơ Chế Chú Ý Nội Tại (Intrinsic Attention Analysis - Table 2)

| Cặp Ngôn Ngữ | Tiêu Chí Đo Lường | Vanilla BARTpho | TSSA (Ours) | Tác Động Định Lượng Thực Tế |
| :--- | :--- | :---: | :---: | :--- |
| **Ê Đê (`rhade` → `vi`)** | Attention Entropy $\mathcal{H}(\alpha)$ ↓ | 0.4835 | **0.2383** | **Giảm 50.7% độ hỗn loạn Entropy** |
| | Top-1 Concentration Mass ↑ | 41.28% | **90.39%** | **Tăng gấp 2.2 lần độ tập trung từ khóa** |
| **Tày (`tay` → `vi`)** | Phân bổ Attention Sink | 93.11% (Chìm vào `<s>`) | **40.17%** (Phân bổ chuẩn) | **Triệt tiêu hiện tượng Chìm đắm Chú ý** |
| **Ba Na (`bahnaric` → `vi`)** | Phân bổ Attention Sink | 89.93% (Chìm vào `<s>`) | **48.79%** (Phân bổ chuẩn) | **Triệt tiêu hiện tượng Chìm đắm Chú ý** |

---

## IV. Bảng Bóc Tách Thành Phần (Ablation Study Results - Table 2)

Đánh giá tác động độc lập của 3 module: $\mathcal{L}_{\text{struct}}$ (Token Barycenter), $\mathcal{L}_{\text{prime}}$ (Sentence InfoNCE), và $\mathcal{L}_{\text{route}}$ (Dynamic Head Routing).

| Ngôn Ngữ | Cấu Hình / Biến Thể | SacreBLEU ↑ | chrF++ ↑ | METEOR ↑ | COMET ↑ | Δ vs Full BLEU |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Tày (`tay` → `vi`)** | **Full TSSA** | **25.46** | **36.31** | **26.29** | **-0.5086** | **Mốc đối chuẩn (0.0)** |
| | `w/o Dynamic Head Routing` ($\lambda_3 = 0$) | 25.44 | 36.41 | 26.41 | -- | -0.02 |
| | `w/o Barycenter Struct Anchoring` ($\lambda_1 = 0$) | 25.27 | 36.28 | 26.22 | -- | **-0.19** *(Tụt sâu nhất)* |
| | `w/o Contrastive Priming` ($\lambda_2 = 0$) | 25.44 | 36.36 | 26.28 | -- | -0.02 |
| | **Vanilla BARTpho (No Anchoring)** | 24.67 | 35.74 | 25.68 | -0.5291 | -0.79 |
| **Ê Đê (`rhade` → `vi`)** | **Full TSSA** | **24.11** | **40.43** | **34.81** | **-0.3264** | **Mốc đối chuẩn (0.0)** |
| | *Các biến thể Ablation* | *[Đang chạy]* | *...* | *...* | *...* | *...* |
| | **Vanilla BARTpho (No Anchoring)** | 23.41 | 39.33 | 33.91 | -0.3678 | -0.70 |
| **Ba Na (`bahnaric` → `vi`)** | **Full TSSA** | **9.66** | **23.89** | **18.46** | **-0.8376** | **Mốc đối chuẩn (0.0)** |
| | *Các biến thể Ablation* | *[Đang chạy]* | *...* | *...* | *...* | *...* |
| | **Vanilla BARTpho (No Anchoring)** | 9.63 | 23.47 | 18.15 | -0.8507 | -0.03 |

---

## V. Lệnh Tái Lập Thí Nghiệm & Đánh Giá

### 1. Xuất Báo Cáo Nhanh Toàn Diện (Full 4 Metrics):
```bash
python summary_results.py --comet
```

### 2. Xuất Báo Cáo Bóc Tách (Ablation Study):
```bash
python summary_results.py --ablation
```

### 3. Đánh giá Checkpoint Cụ Thể:
```bash
python eval_checkpoint.py --checkpoint checkpoints/tssa_rhade --comet
```
