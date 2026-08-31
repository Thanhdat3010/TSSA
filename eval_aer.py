"""
Alignment Error Rate (AER) & Attention Entropy Evaluation Script (TSSA)
Evaluates intrinsic alignment quality of Transformer models:
1. Alignment Error Rate (AER % down)
2. Alignment Precision (P % up), Recall (R % up), F1 (% up)
3. Cross-Attention Entropy (H(alpha) down - Attention Sharpness)

References:
- Och & Ney (Computational Linguistics 2003): AER definition
- Jalili Sabet et al. (EMNLP 2020): SimAlign Silver Standard
- Voita et al. (ACL 2019): Analyzing Multi-Head Self-Attention
"""

import os
import argparse
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from models.tssa_seq2seq import TSSASeq2SeqModel

def compute_simalign_silver(src_tokens, tgt_tokens, simaligner=None):
    """Computes Silver standard word alignments using SimAlign or token heuristic."""
    align_set = set()
    if simaligner is not None and len(src_tokens) > 0 and len(tgt_tokens) > 0:
        try:
            res = simaligner.get_word_aligns(src_tokens, tgt_tokens)
            for (s, t) in res.get("itermax", res.get("inter", [])):
                align_set.add((t, s)) # Target-to-Source (t, s)
            return align_set
        except Exception:
            pass

    # Exact or subword match fallback
    for s_idx, s_tok in enumerate(src_tokens):
        for t_idx, t_tok in enumerate(tgt_tokens):
            if s_tok.lower() == t_tok.lower() and len(s_tok) > 1:
                align_set.add((t_idx, s_idx))
            elif abs(s_idx / max(1, len(src_tokens)) - t_idx / max(1, len(tgt_tokens))) < 0.08:
                align_set.add((t_idx, s_idx))
    return align_set

def evaluate_alignment_metrics(checkpoint_dir: str, lang: str, data_dir: str = "data_processed",
                               batch_size: int = 16, max_samples: int = 500, device: str = "cuda"):
    """
    Computes AER, F1, Precision, Recall and Attention Entropy on the test set.
    """
    test_csv = os.path.join(data_dir, lang, "test.csv")
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Không tìm thấy {test_csv}")

    df = pd.read_csv(test_csv).head(max_samples)
    print(f"\n============================================================")
    print(f"[*] Đánh giá AER & Attention Entropy cho: {checkpoint_dir} ({lang})")
    print(f"[*] Số lượng mẫu test: {len(df)}")
    print(f"============================================================")

    # 1. Nạp Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-syllable")
    
    # Try loading as TSSASeq2SeqModel first, fallback to AutoModelForSeq2SeqLM
    try:
        model = TSSASeq2SeqModel.from_pretrained(checkpoint_dir) if hasattr(TSSASeq2SeqModel, "from_pretrained") else None
    except Exception:
        model = None

    if model is None:
        try:
            model = TSSASeq2SeqModel(model_name_or_path=checkpoint_dir).to(device)
        except Exception:
            model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_dir).to(device)

    model.eval()

    # 2. Khởi tạo SimAlign (Silver Reference Aligner)
    simaligner = None
    try:
        from simalign import SentenceAligner
        simaligner = SentenceAligner(model_name_or_path="vinai/bartpho-syllable", token_type="bpe", device=device)
        print("[+] Đã nạp SimAlign Silver Aligner!")
    except Exception as e:
        print(f"[!] Warning: SimAlign không khả dụng ({e}). Sử dụng Heuristic Matcher.")

    total_intersection = 0
    total_pred = 0
    total_gold = 0
    entropy_list = []

    print("[*] Đang trích xuất ma trận Cross-Attention và chấm điểm AER...")
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc="AER Eval"):
            src_text = str(row["src_text"])
            tgt_text = str(row["tgt_text"])

            src_enc = tokenizer(src_text, max_length=128, truncation=True, return_tensors="pt").to(device)
            with tokenizer.as_target_tokenizer():
                tgt_enc = tokenizer(tgt_text, max_length=128, truncation=True, return_tensors="pt").to(device)

            src_tokens = tokenizer.tokenize(src_text)[:128]
            tgt_tokens = tokenizer.tokenize(tgt_text)[:128]
            T_len = len(tgt_tokens)
            S_len = len(src_tokens)

            if T_len == 0 or S_len == 0:
                continue

            # 1. Gold / Silver Alignments G
            gold_set = compute_simalign_silver(src_tokens, tgt_tokens, simaligner)
            if len(gold_set) == 0:
                continue

            # 2. Extract Cross-Attention from Model
            outputs = model(
                input_ids=src_enc["input_ids"],
                attention_mask=src_enc["attention_mask"],
                labels=tgt_enc["input_ids"],
                output_attentions=True,
                output_hidden_states=True
            )

            # Get cross attention from top layer
            cross_attns = outputs.get("cross_attentions") if isinstance(outputs, dict) else getattr(outputs, "cross_attentions", None)
            
            if cross_attns is not None and len(cross_attns) > 0 and cross_attns[-1] is not None:
                attn_map = cross_attns[-1][0].mean(dim=0) # Average over heads: [T, S]
            else:
                # Direct QK calculation if cross_attns was None in SDPA
                dec_states = outputs["decoder_hidden_states"] if isinstance(outputs, dict) else outputs.decoder_hidden_states
                enc_state = outputs["encoder_last_hidden_state"] if isinstance(outputs, dict) else outputs.encoder_last_hidden_state
                dec_state = dec_states[-1] # [1, T, D]
                
                # Use last decoder layer
                inner = getattr(model, "model", model)
                dec_layer = inner.model.decoder.layers[-1]
                
                D = dec_state.size(-1)
                H = 16
                d_k = D // H
                q = dec_layer.encoder_attn.q_proj(dec_state).view(1, -1, H, d_k).transpose(1, 2)
                k = dec_layer.encoder_attn.k_proj(enc_state).view(1, -1, H, d_k).transpose(1, 2)
                scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
                attn_map = F.softmax(scores, dim=-1)[0].mean(dim=0) # [T, S]

            # Slice to active tokens
            attn_cut = attn_map[:T_len, :S_len].cpu() # [T, S]

            # 3. Compute Attention Entropy: H(alpha) = - sum alpha * log(alpha + eps)
            eps = 1e-8
            ent = - (attn_cut * torch.log(attn_cut + eps)).sum(dim=-1).mean().item()
            entropy_list.append(ent)

            # 4. Predict Alignment Set A (Argmax per target token)
            pred_set = set()
            max_indices = torch.argmax(attn_cut, dim=-1).tolist()
            for t_idx, s_idx in enumerate(max_indices):
                pred_set.add((t_idx, s_idx))

            # Cumulative stats for AER
            inter = len(pred_set.intersection(gold_set))
            total_intersection += inter
            total_pred += len(pred_set)
            total_gold += len(gold_set)

    precision = (total_intersection / max(1, total_pred)) * 100.0
    recall = (total_intersection / max(1, total_gold)) * 100.0
    f1 = (2 * precision * recall) / max(1e-5, (precision + recall))
    aer = (1.0 - (2.0 * total_intersection) / max(1, (total_pred + total_gold))) * 100.0
    avg_entropy = np.mean(entropy_list) if len(entropy_list) > 0 else 0.0

    print("\n================ KẾT QUẢ ĐÁNH GIÁ CĂN CHỈNH (INTRINSIC ALIGNMENT) ================")
    print(f"  Alignment Precision (P) : {precision:.2f}%  ↑")
    print(f"  Alignment Recall (R)    : {recall:.2f}%  ↑")
    print(f"  Alignment F1 Score (F1) : {f1:.2f}%  ↑")
    print(f"  Alignment Error Rate (AER): {aer:.2f}%  ↓ (Càng thấp càng tốt)")
    print(f"  Attention Entropy H(α)  : {avg_entropy:.4f}  ↓ (Càng thấp độ tập trung càng cao)")
    print("==================================================================================")

    return {
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "f1": round(f1, 2),
        "aer": round(aer, 2),
        "entropy": round(avg_entropy, 4)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate AER and Attention Entropy for TSSA models.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint directory.")
    parser.add_argument("--lang", type=str, required=True, choices=["rhade", "bahnaric", "tay"], help="Language code.")
    parser.add_argument("--max_samples", type=int, default=500, help="Maximum number of test samples.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluate_alignment_metrics(args.checkpoint, args.lang, max_samples=args.max_samples, device=device)
