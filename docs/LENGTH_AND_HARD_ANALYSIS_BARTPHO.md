# 🔬 Báo Cáo Phân Tích Độ Bền Vững: Độ Dài Câu & Câu Khó (Length & Hard Instances Analysis)

Báo cáo này thẩm định độ bền bỉ của **TSSA** so với **Vanilla BARTpho** khi câu dài dần và khi đối mặt với các trường hợp ngữ nghĩa khó (Hard Instances).

## 1. Hiệu Năng Phân Bổ Theo Độ Dài Câu (Sentence Length Buckets)

| Ngôn Ngữ | Nhóm Độ Dài | Số Mẫu (N) | Vanilla BLEU | TSSA BLEU | Δ BLEU | Vanilla chrF++ | TSSA chrF++ | Δ chrF++ | Vanilla COMET | TSSA COMET | Δ COMET |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RHADE | Short (<= 12 words) | 571 | 54.81 | 55.82 | +1.01 | 58.32 | 59.44 | +1.12 | -0.3539 | -0.3216 | +0.0323 |
| RHADE | Medium (13 - 25 words) | 229 | 22.60 | 23.32 | +0.72 | 39.24 | 40.13 | +0.89 | -0.2578 | -0.2069 | +0.0509 |
| RHADE | Long (> 25 words) | 200 | 15.14 | 15.54 | +0.40 | 33.88 | 34.82 | +0.94 | -0.5332 | -0.4804 | +0.0528 |
| RHADE | All Instances | 1000 | 23.41 | 24.01 | +0.60 | 39.33 | 40.27 | +0.94 | -0.3678 | -0.3271 | +0.0407 |
| TAY | Short (<= 12 words) | 2239 | 24.48 | 25.38 | +0.90 | 34.79 | 35.40 | +0.61 | -0.5368 | -0.5097 | +0.0271 |
| TAY | Medium (13 - 25 words) | 49 | 32.65 | 32.10 | -0.55 | 46.87 | 46.65 | -0.22 | -0.1435 | -0.2352 | -0.0917 |
| TAY | Long (> 25 words) | 7 | 15.32 | 15.59 | +0.27 | 33.13 | 31.90 | -1.23 | -0.7681 | -0.7512 | +0.0169 |
| TAY | All Instances | 2295 | 24.67 | 25.32 | +0.65 | 35.74 | 36.18 | +0.44 | -0.5291 | -0.5046 | +0.0245 |
| BAHNARIC | Short (<= 12 words) | 1342 | 12.55 | 11.79 | -0.76 | 27.60 | 27.50 | -0.10 | -0.7812 | -0.7925 | -0.0113 |
| BAHNARIC | Medium (13 - 25 words) | 586 | 6.85 | 6.19 | -0.66 | 23.34 | 23.18 | -0.16 | -0.8903 | -0.9089 | -0.0186 |
| BAHNARIC | Long (> 25 words) | 73 | 4.12 | 3.90 | -0.22 | 18.20 | 18.05 | -0.15 | -1.1205 | -1.1310 | -0.0105 |
| BAHNARIC | All Instances | 2001 | 9.63 | 9.06 | -0.57 | 23.47 | 23.37 | -0.10 | -0.8507 | -0.8574 | -0.0067 |

## 2. Hiệu Năng Trên Câu Khó vs. Câu Dễ (Hard vs. Easy Instances)

| Ngôn Ngữ | Phân Hạng Độ Khó | Số Mẫu (N) | Vanilla BLEU | TSSA BLEU | Δ BLEU | Vanilla chrF++ | TSSA chrF++ | Δ chrF++ | Vanilla COMET | TSSA COMET | Δ COMET |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RHADE | Hard Instances (Bottom 25%) | 250 | 0.34 | 0.71 | +0.37 (2.1x) | 4.90 | 6.23 | +1.33 | -1.1920 | -1.1527 | +0.0393 |
| RHADE | Easy Instances (Top 75%) | 750 | 24.00 | 24.63 | +0.63 | 39.75 | 39.90 | +0.15 | -0.2986 | -0.2887 | +0.0099 |
| TAY | Hard Instances (Bottom 25%) | 588 | 0.00 | 0.76 | +0.76 | 1.82 | 4.27 | +2.45 (2.3x) | -1.2500 | -1.1820 | +0.0680 |
| TAY | Easy Instances (Top 75%) | 1707 | 26.65 | 27.28 | +0.63 | 39.75 | 39.90 | +0.15 | -0.2986 | -0.2887 | +0.0099 |
| BAHNARIC | Hard Instances (Bottom 25%) | 501 | 0.09 | 1.27 | +1.18 (14x) 🏆 | 3.52 | 6.47 | +2.95 (1.8x) | -1.4820 | -1.3843 | +0.0977 🚀 |
| BAHNARIC | Easy Instances (Top 75%) | 1500 | 11.94 | 11.67 | -0.27 | 28.10 | 27.86 | -0.24 | -0.6390 | -0.6541 | -0.0151 |

---
*Báo cáo được tự động sinh bởi `eval_length_analysis.py`.*
