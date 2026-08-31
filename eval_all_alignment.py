"""
Batch Alignment & AER Evaluator for All Checkpoints (TSSA)
Evaluates and outputs a complete comparative benchmark table:
Baseline vs TSSA 2.0 across all 3 ethnic minority languages (Rhade, Bahnaric, Tay).

Metrics:
1. Alignment Error Rate (AER % down - lower is better)
2. Alignment F1 Score (F1 % up - higher is better)
3. Cross-Attention Entropy H(alpha) down (lower is sharper)
4. Top-1 Attention Concentration % up
"""

import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from models.tssa_seq2seq import TSSASeq2SeqModel

def find_model_weights_dir(path: str) -> str:
    """Finds if path or subfolder contains model weight files."""
    if not os.path.exists(path) or not os.path.isdir(path):
        return None
    files = os.listdir(path)
    for f in files:
        if f.endswith(".safetensors") or f.endswith(".bin") or f == "pytorch_model.bin":
            return path
    subdirs = [os.path.join(path, d) for d in files if os.path.isdir(os.path.join(path, d))]
    for s in sorted(subdirs, reverse=True):
        try:
            for f in os.listdir(s):
                if f.endswith(".safetensors") or f.endswith(".bin") or f == "pytorch_model.bin":
                    return s
        except Exception:
            pass
    return None

def compute_neural_silver_alignments(src_enc_states: torch.Tensor, tgt_enc_states: torch.Tensor, threshold: float = 0.15) -> set:
    """Computes Silver standard cross-lingual word alignments using Bidirectional IterMax."""
    src_norm = F.normalize(src_enc_states, p=2, dim=-1)
    tgt_norm = F.normalize(tgt_enc_states, p=2, dim=-1)
    sim = torch.mm(tgt_norm, src_norm.t()) # [T, S]

    t_to_s = torch.argmax(sim, dim=-1) # [T]
    s_to_t = torch.argmax(sim, dim=0)  # [S]

    align_set = set()
    for t_idx in range(sim.size(0)):
        s_idx = t_to_s[t_idx].item()
        if s_to_t[s_idx].item() == t_idx and sim[t_idx, s_idx].item() >= threshold:
            align_set.add((t_idx, s_idx))
            
    for t_idx in range(sim.size(0)):
        s_idx = t_to_s[t_idx].item()
        if sim[t_idx, s_idx].item() >= 0.25:
            align_set.add((t_idx, s_idx))

    return align_set

def evaluate_single_checkpoint(checkpoint_dir: str, lang: str, fallback_model: str = "vinai/bartpho-syllable",
                               data_dir: str = "data_processed", max_samples: int = 300, device: str = "cuda"):
    test_csv = os.path.join(data_dir, lang, "test.csv")
    if not os.path.exists(test_csv):
        return None

    df = pd.read_csv(test_csv).head(max_samples)
    tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-syllable")
    
    resolved_path = find_model_weights_dir(checkpoint_dir)
    if resolved_path is not None:
        load_path = resolved_path
    else:
        if fallback_model is not None:
            load_path = fallback_model
        else:
            return None

    try:
        model = TSSASeq2SeqModel(model_name_or_path=load_path).to(device)
    except Exception:
        model = AutoModelForSeq2SeqLM.from_pretrained(load_path).to(device)

    model.eval()
    entropy_list = []
    top1_mass_list = []
    total_inter = 0
    total_pred = 0
    total_gold = 0

    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Eval {os.path.basename(checkpoint_dir)}", leave=False):
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

            outputs = model(
                input_ids=src_enc["input_ids"],
                attention_mask=src_enc["attention_mask"],
                labels=tgt_enc["input_ids"],
                output_hidden_states=True
            )

            dec_states = outputs["decoder_hidden_states"] if isinstance(outputs, dict) else outputs.decoder_hidden_states
            enc_states = outputs["encoder_last_hidden_state"] if isinstance(outputs, dict) else outputs.encoder_last_hidden_state
            dec_state = dec_states[-1]
            
            inner_model = getattr(model, "model", model)
            dec_layer = inner_model.model.decoder.layers[-1]
            
            D = dec_state.size(-1)
            H = 16
            d_k = D // H
            q = dec_layer.encoder_attn.q_proj(dec_state).view(1, -1, H, d_k).transpose(1, 2)
            k = dec_layer.encoder_attn.k_proj(enc_states).view(1, -1, H, d_k).transpose(1, 2)
            scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
            attn_map = F.softmax(scores, dim=-1)[0].mean(dim=0) # [T, S]

            attn_cut = attn_map[:T_len, :S_len].cpu()

            # 1. Entropy & Top-1 Concentration
            eps = 1e-8
            ent = - (attn_cut * torch.log(attn_cut + eps)).sum(dim=-1).mean().item()
            entropy_list.append(ent)

            top1 = attn_cut.max(dim=-1).values.mean().item() * 100.0
            top1_mass_list.append(top1)

            # 2. Silver Ground Truth & AER Computation
            tgt_enc_out = inner_model.model.encoder(
                input_ids=tgt_enc["input_ids"],
                attention_mask=tgt_enc["attention_mask"],
                return_dict=True
            )
            tgt_states = tgt_enc_out.last_hidden_state[0, :T_len]
            src_states = enc_states[0, :S_len]

            gold_set = compute_neural_silver_alignments(src_states, tgt_states, threshold=0.15)
            if len(gold_set) > 0:
                pred_set = set()
                max_idx = torch.argmax(attn_cut, dim=-1).tolist()
                for t_i, s_i in enumerate(max_idx):
                    pred_set.add((t_i, s_i))

                inter = len(pred_set.intersection(gold_set))
                total_inter += inter
                total_pred += len(pred_set)
                total_gold += len(gold_set)

    avg_entropy = np.mean(entropy_list) if len(entropy_list) > 0 else 0.0
    avg_top1 = np.mean(top1_mass_list) if len(top1_mass_list) > 0 else 0.0

    precision = (total_inter / max(1, total_pred)) * 100.0
    recall = (total_inter / max(1, total_gold)) * 100.0
    f1 = (2 * precision * recall) / max(1e-5, (precision + recall))
    aer = (1.0 - (2.0 * total_inter) / max(1, (total_pred + total_gold))) * 100.0

    return {
        "aer": round(aer, 2),
        "f1": round(f1, 2),
        "entropy": round(avg_entropy, 4),
        "top1_concentration": round(avg_top1, 2)
    }

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print("  🚀 ĐANG ĐÁNH GIÁ ĐỒNG THỜI CẢ AER (ALIGNMENT ERROR RATE) VÀ ATTENTION ENTROPY")
    print("=" * 80)

    configs = [
        ("Ê Đê (rhade)", "rhade", "checkpoints/bartpho_vanilla_rhade", "checkpoints/tssa_rhade"),
        ("Tày (tay)", "tay", "checkpoints/bartpho_vanilla_tay", "checkpoints/tssa_tay"),
        ("Ba Na (bahnaric)", "bahnaric", "checkpoints/bartpho_vanilla_bahnaric", "checkpoints/tssa_bahnaric")
    ]

    results = []
    for lang_name, lang_code, base_dir, tssa_dir in configs:
        print(f"\n[*] Đang đánh giá cặp: {lang_name} ...")
        base_res = evaluate_single_checkpoint(base_dir, lang_code, fallback_model="vinai/bartpho-syllable", device=device)
        tssa_res = evaluate_single_checkpoint(tssa_dir, lang_code, fallback_model=None, device=device)

        results.append({
            "lang": lang_name,
            "base_aer": f"{base_res['aer']}%" if base_res else "--",
            "tssa_aer": f"{tssa_res['aer']}%" if tssa_res else "--",
            "base_f1": f"{base_res['f1']}%" if base_res else "--",
            "tssa_f1": f"{tssa_res['f1']}%" if tssa_res else "--",
            "base_ent": base_res["entropy"] if base_res else "--",
            "tssa_ent": tssa_res["entropy"] if tssa_res else "--",
            "base_top1": f"{base_res['top1_concentration']}%" if base_res else "--",
            "tssa_top1": f"{tssa_res['top1_concentration']}%" if tssa_res else "--"
        })

    print("\n" + "=" * 105)
    print("📊 BẢNG 3: SO SÁNH CHẤT LƯỢNG CĂN CHỈNH NỘI TẠI (AER, F1 & ATTENTION ENTROPY)")
    print("=" * 105)
    print(f"| {'Ngôn Ngữ':<18} | {'Base AER ↓':<12} | {'TSSA AER ↓':<12} | {'Base F1 ↑':<11} | {'TSSA F1 ↑':<11} | {'Base H(α) ↓':<13} | {'TSSA H(α) ↓':<13} |")
    print(f"|{'-'*20}|{'-'*14}|{'-'*14}|{'-'*13}|{'-'*13}|{'-'*15}|{'-'*15}|")
    for r in results:
        print(f"| {r['lang']:<18} | {str(r['base_aer']):<12} | {str(r['tssa_aer']):<12} | {str(r['base_f1']):<11} | {str(r['tssa_f1']):<11} | {str(r['base_ent']):<13} | {str(r['tssa_ent']):<13} |")
    print("=" * 105)

if __name__ == "__main__":
    main()
