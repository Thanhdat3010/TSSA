# TSSA: Target-Side Semantic Anchoring for Low-Resource Machine Translation

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Official PyTorch implementation for the paper: **"TSSA: Target-Side Semantic Anchoring with Gated Multi-Head Cross-Attention for Extremely Low-Resource Ethnic Minority Machine Translation"**.

---

## 📖 Overview

Low-resource Machine Translation (NMT) for ethnic minority languages (Bahnar, Rhade/Ê Đê, Tay) into high-resource languages (Vietnamese) often suffers from representation degradation and cross-lingual semantic misalignment. 

**TSSA** introduces an asymmetric semantic anchoring framework:
1. **Confidence-Weighted Barycenter Anchoring ($\mathcal{L}_{\text{struct}}$):** Projects minority source subword tokens into the target semantic space using Stop-Gradient $\text{sg}(\cdot)$ frozen teacher representations.
2. **In-Batch Sentence InfoNCE Priming ($\mathcal{L}_{\text{prime}}$):** Enforces global semantic coherence between source and target sentence embeddings.
3. **Decoder Head-Wise Router ($\mathcal{L}_{\text{route}}$):** Dynamically gates cross-attention heads using an MLP to route information specifically through reliable semantic anchor heads.

---

## 🗂️ Supported Datasets

| Language Pair | Source Language | Target Language | Official Splits | Source Repo |
| :--- | :--- | :--- | :---: | :--- |
| **Bahnaric – Vietnamese** | Ba Na (`bahnaric`) | Tiếng Việt (`vietnamese`) | 51.9k train / test | `FiveC/bahnaric_vietnamese` |
| **Rhade – Vietnamese** | Ê Đê (`ede` / `cdc`) | Tiếng Việt (`vi`) | 15.1k train / test | `NIRVLab/rhade-vietnamese-mt` |
| **Tay – Vietnamese** | Tày (`tay`) | Tiếng Việt (`viet`) | 20.6k train / val $\rightarrow$ test | `HeyDunaX/tay-vietnamese-nmt` |

---

## 🛠️ Installation & Environment Setup

### 1. Conda Environment Setup:
```bash
conda create -n TSSA python=3.10 -y
conda activate TSSA
```

### 2. Install Dependencies:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### 1. Download and Preprocess Datasets
Downloads official splits from Hugging Face, unzips translation dicts, normalizes Unicode NFC, and generates leak-free `train.csv` and `test.csv`:
```bash
python data/download_and_preprocess.py
```

### 2. Train TSSA Model (Proposed)
```bash
# Train on Rhade (Ê Đê)
python train.py --lang rhade --model_type tssa --max_source_length 256 --max_target_length 256

# Train on Tay (Tày)
python train.py --lang tay --model_type tssa --max_source_length 256 --max_target_length 256

# Train on Bahnaric (Ba Na)
python train.py --lang bahnaric --model_type tssa --max_source_length 256 --max_target_length 256
```

### 3. Train Competitor Baselines
All competitor methods run on the **identical `BARTpho` backbone** for fair comparison:
```bash
# Guided Cross-Attention (Chen et al., ACL 2016)
python train.py --lang rhade --model_type guided_attn

# Joint-Align (Garg et al., EMNLP 2019)
python train.py --lang rhade --model_type joint_align

# AWESOME-align Loss (Dou & Neubig, EACL 2021)
python train.py --lang rhade --model_type awesome_align

# Cross-Lingual InfoNCE (ACL 2024)
python train.py --lang rhade --model_type cl_lsa

# Vanilla BARTpho Fine-tuning
python train.py --lang rhade --model_type bartpho_vanilla
```

### 4. Evaluate & Run Mechanistic Ablations
```bash
# Standard Benchmark Evaluation (BLEU, chrF++, METEOR, COMET)
python evaluate.py --checkpoint_dir checkpoints/tssa_rhade --lang rhade

# Ablation 3: Causal Head-Pruning (Top-K vs Random-K vs Bottom-K)
python evaluate.py --checkpoint_dir checkpoints/tssa_rhade --lang rhade --run_causal_pruning

# Ablation 4: Robustness against 4 types of Noise
python evaluate.py --checkpoint_dir checkpoints/tssa_rhade --lang rhade --run_robustness
```

---

## 📁 Repository Structure

```text
├── configs/                     # YAML configuration files for experiments and ablations
├── data/                        # Dataset downloading, preprocessing, SimAlign and DataLoaders
├── models/                      # PyTorch models (TSSASeq2Seq, HeadWiseRouter, TeacherWrapper)
├── losses/                      # Mathematical Loss functions (StructLoss, PrimeLoss, RouteLoss, Baselines)
├── training/                    # 3-Phase LossScheduler, TSSASeq2SeqTrainer, Optimizer
├── evaluation/                  # Multi-metric evaluator, Causal Head Pruning, Robustness noise
├── docs/                        # Complete scientific specifications and experiment blueprints
├── requirements.txt             # Python dependencies
├── test_suite.py                # Unit & integration test suite
├── train.py                     # Unified training entrypoint
└── evaluate.py                  # Standalone evaluation entrypoint
```

---

## 📜 Citation

If you use this codebase or method in your research, please cite:
```bibtex
@inproceedings{tssa2026,
  title={TSSA: Target-Side Semantic Anchoring with Gated Multi-Head Cross-Attention for Extremely Low-Resource Ethnic Minority Machine Translation},
  author={Anonymous Authors},
  booktitle={ACL / EMNLP},
  year={2026}
}
```
