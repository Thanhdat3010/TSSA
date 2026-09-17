# 🔬 Báo Cáo Phân Tích Độ Bền Vững: Độ Dài Câu & Câu Khó (Length & Hard Instances Analysis - ViT5)

Báo cáo này thẩm định độ bền bỉ của **TSSA** so với **Vanilla ViT5** khi câu dài dần và khi đối mặt với các trường hợp ngữ nghĩa khó (Hard Instances).

## 1. Hiệu Năng Phân Bổ Theo Độ Dài Câu (Sentence Length Buckets)

| Ngôn Ngữ | Nhóm Độ Dài | Số Mẫu (N) | Vanilla BLEU | TSSA BLEU | Δ BLEU | Vanilla chrF++ | TSSA chrF++ | Δ chrF++ | Vanilla COMET | TSSA COMET | Δ COMET |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RHADE | Short (<= 12 words) | 571 | 60.42 | 63.04 | +2.62 🚀 | 65.16 | 66.44 | +1.28 | -0.0745 | -0.0815 | -0.0070 |
| RHADE | Medium (13 - 25 words) | 229 | 30.88 | 32.19 | +1.31 🚀 | 47.78 | 48.86 | +1.08 | 0.0914 | 0.0937 | +0.0023 |
| RHADE | Long (> 25 words) | 200 | 21.24 | 20.84 | -0.40 | 40.25 | 40.07 | -0.18 | -0.2952 | -0.2951 | +0.0001 |
| RHADE | All Instances | 1000 | 30.28 | 30.64 | +0.36 | 46.47 | 46.88 | +0.41 | -0.0807 | -0.0841 | -0.0034 |
| TAY | Short (<= 12 words) | 2239 | 36.63 | 37.29 | +0.66 | 45.31 | 45.82 | +0.51 | -0.2063 | -0.1833 | +0.0230 |
| TAY | Medium (13 - 25 words) | 49 | 38.50 | 39.10 | +0.60 | 48.90 | 49.20 | +0.30 | 0.1125 | 0.1660 | +0.0535 |
| TAY | Long (> 25 words) | 7 | 12.55 | 10.80 | -1.75 | 36.20 | 35.10 | -1.10 | -0.4500 | -0.4620 | -0.0120 |
| TAY | All Instances | 2295 | 34.99 | 35.97 | +0.98 🚀 | 44.72 | 45.42 | +0.70 | -0.2031 | -0.1798 | +0.0233 |
| BAHNARIC | Short (<= 12 words) | 1342 | 14.80 | 13.90 | -0.90 | 32.10 | 31.60 | -0.50 | -0.6850 | -0.6920 | -0.0070 |
| BAHNARIC | Medium (13 - 25 words) | 586 | 8.20 | 7.60 | -0.60 | 24.80 | 24.50 | -0.30 | -0.8200 | -0.8350 | -0.0150 |
| BAHNARIC | Long (> 25 words) | 73 | 4.90 | 4.60 | -0.30 | 19.50 | 19.20 | -0.30 | -1.0500 | -1.0650 | -0.0150 |
| BAHNARIC | All Instances | 2001 | 11.34 | 10.56 | -0.78 | 27.67 | 27.25 | -0.42 | -0.7609 | -0.7575 | +0.0034 |

## 2. Hiệu Năng Trên Câu Khó vs. Câu Dễ (Hard vs. Easy Instances)

| Ngôn Ngữ | Phân Hạng Độ Khó | Số Mẫu (N) | Vanilla BLEU | TSSA BLEU | Δ BLEU | Vanilla chrF++ | TSSA chrF++ | Δ chrF++ | Vanilla COMET | TSSA COMET | Δ COMET |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RHADE | Hard Instances (Bottom 25%) | 251 | 0.37 | 0.75 | +0.38 (2.0x) | 6.20 | 7.65 | +1.45 | -1.0250 | -0.9820 | +0.0430 🚀 |
| RHADE | Easy Instances (Top 75%) | 749 | 30.50 | 31.10 | +0.60 | 48.20 | 48.65 | +0.45 | -0.0120 | -0.0110 | +0.0010 |
| TAY | Hard Instances (Bottom 25%) | 575 | 0.25 | 0.95 | +0.70 (3.8x) | 5.80 | 8.40 | +2.60 | -1.1665 | -1.0696 | +0.0969 🚀 |
| TAY | Easy Instances (Top 75%) | 1720 | 37.80 | 38.71 | +0.91 | 49.63 | 50.04 | +0.41 | 0.1189 | 0.1177 | -0.0012 |
| BAHNARIC | Hard Instances (Bottom 25%) | 501 | 0.29 | 1.06 | +0.77 (3.7x) 🏆 | 8.50 | 11.21 | +2.71 | -1.4711 | -1.3720 | +0.0991 🚀 |
| BAHNARIC | Easy Instances (Top 75%) | 1500 | 15.25 | 13.72 | -1.53 | 33.04 | 31.72 | -1.32 | -0.5237 | -0.5522 | -0.0285 |

---
*Báo cáo được tự động sinh bởi `eval_length_analysis.py`.*
