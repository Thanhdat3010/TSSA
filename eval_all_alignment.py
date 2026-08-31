"""
Batch Alignment & Attention Entropy Evaluator for All Checkpoints (TSSA)
Evaluates and outputs a complete comparative benchmark table:
Baseline vs TSSA 2.0 across all 3 ethnic minority languages (Rhade, Bahnaric, Tay).
"""

import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from models.tssa_seq2seq import TSSASeq2SeqModel

def evaluate_single_checkpoint(checkpoint_dir: str, lang: str, data_dir: str = "data_processed",
                               max_samples: int = 300, device: str = "cuda"):
    test_csv = os.path.join(data_dir, lang, "test.csv")
    if not os.path.exists(test_csv) or not os.path.exists(checkpoint_dir):
        return None

    df = pd.read_csv(test_csv).head(max_samples)
    tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-syllable")
    
    try:
        model = TSSASeq2SeqModel(model_name_or_path=checkpoint_dir).to(device)
    except Exception:
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_dir).to(device)

    model.eval()
    entropy_list = []
    top1_mass_list = []

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

            # Compute cross attention map
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

            # 1. Attention Entropy: H(alpha)
            eps = 1e-8
            ent = - (attn_cut * torch.log(attn_cut + eps)).sum(dim=-1).mean().item()
            entropy_list.append(ent)

            # 2. Top-1 Attention Concentration: max_s alpha_{t, s}
            top1 = attn_cut.max(dim=-1).values.mean().item() * 100.0
            top1_mass_list.append(top1)

    avg_entropy = np.mean(entropy_list) if len(entropy_list) > 0 else 0.0
    avg_top1 = np.mean(top1_mass_list) if len(top1_mass_list) > 0 else 0.0

    return {
        "entropy": round(avg_entropy, 4),
        "top1_concentration": round(avg_top1, 2)
    }

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print("  🚀 ĐANG ĐÁNH GIÁ CHỈ SỐ CĂN CHỈNH VÀ ATTENTION ENTROPY CHO CẢ 3 NGÔN NGỮ")
    print("=" * 70)

    configs = [
        ("Ê Đê (rhade)", "rhade", "checkpoints/bartpho_vanilla_rhade", "checkpoints/tssa_rhade"),
        ("Tày (tay)", "tay", "checkpoints/bartpho_vanilla_tay", "checkpoints/tssa_tay"),
        ("Ba Na (bahnaric)", "bahnaric", "checkpoints/bartpho_vanilla_bahnaric", "checkpoints/tssa_bahnaric")
    ]

    results = []
    for lang_name, lang_code, base_dir, tssa_dir in configs:
        print(f"\n[*] Đang đánh giá cặp: {lang_name} ...")
        base_res = evaluate_single_checkpoint(base_dir, lang_code, device=device)
        tssa_res = evaluate_single_checkpoint(tssa_dir, lang_code, device=device)

        results.append({
            "lang": lang_name,
            "base_ent": base_res["entropy"] if base_res else "--",
            "tssa_ent": tssa_res["entropy"] if tssa_res else "--",
            "base_top1": f"{base_res['top1_concentration']}%" if base_res else "--",
            "tssa_top1": f"{tssa_res['top1_concentration']}%" if tssa_res else "--"
        })

    print("\n" + "=" * 80)
    print("📊 BẢNG 3: SO SÁNH ĐỘ TẬP TRUNG CHÚ Ý (ATTENTION SHARPNESS & CONCENTRATION)")
    print("=" * 80)
    print(f"| {'Ngôn Ngữ':<18} | {'Baseline Entropy H(α) ↓':<24} | {'TSSA 2.0 Entropy H(α) ↓':<24} | {'Baseline Top-1 Mass ↑':<22} | {'TSSA 2.0 Top-1 Mass ↑':<22} |")
    print(f"|{'-'*20}|{'-'*26}|{'-'*26}|{'-'*24}|{'-'*24}|")
    for r in results:
        print(f"| {r['lang']:<18} | {str(r['base_ent']):<24} | {str(r['tssa_ent']):<24} | {str(r['base_top1']):<22} | {str(r['tssa_top1']):<22} |")
    print("=" * 80)

if __name__ == "__main__":
    main()
