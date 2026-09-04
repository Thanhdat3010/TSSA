"""
Cross-Attention Heatmap & Attention Sink Elimination Visualizer
Generates publication-quality side-by-side heatmaps comparing:
(a) Vanilla BARTpho (Severe Attention Sinking on <s> delimiter token)
(b) TSSA (Ours) (Sharp, diagonal semantic alignment with Attention Sink eliminated)

Saves both high-resolution PNG (300 DPI) and vector PDF in docs/figures/ for paper integration.
"""

import os
import argparse
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np

# Matplotlib setup with safe headless backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from models.tssa_seq2seq import TSSASeq2SeqModel

def extract_cross_attention(model, tokenizer, src_text, tgt_text, device="cpu", layer_idx=-1):
    """
    Runs forward pass and extracts cross-attention matrix [T, S] from the specified decoder layer.
    """
    model.eval()
    src_enc = tokenizer(src_text, max_length=64, truncation=True, return_tensors="pt").to(device)
    with tokenizer.as_target_tokenizer():
        tgt_enc = tokenizer(tgt_text, max_length=64, truncation=True, return_tensors="pt").to(device)

    # Subword token lists for axis labels
    src_tokens = tokenizer.convert_ids_to_tokens(src_enc["input_ids"][0])
    tgt_tokens = tokenizer.convert_ids_to_tokens(tgt_enc["input_ids"][0])
    
    # Clean token strings for display
    src_labels = [t.replace("@@", "").replace(" ", "") if t != " " else " " for t in src_tokens]
    tgt_labels = [t.replace("@@", "").replace(" ", "") if t != " " else " " for t in tgt_tokens]

    with torch.no_grad():
        outputs = model(
            input_ids=src_enc["input_ids"],
            attention_mask=src_enc["attention_mask"],
            labels=tgt_enc["input_ids"],
            output_attentions=True,
            output_hidden_states=True
        )

        # Trích xuất cross-attention tuple
        cross_attns = outputs.get("cross_attentions") if isinstance(outputs, dict) else getattr(outputs, "cross_attentions", None)

        if cross_attns is not None and len(cross_attns) > 0 and cross_attns[layer_idx] is not None:
            # cross_attns[layer_idx] có shape [B, H, T, S]
            attn_layer = cross_attns[layer_idx][0] # [H, T, S]
            attn_map = attn_layer.mean(dim=0).cpu().numpy() # [T, S]
        else:
            # Fallback tính qua Q, K
            dec_states = outputs["decoder_hidden_states"] if isinstance(outputs, dict) else outputs.decoder_hidden_states
            enc_states = outputs["encoder_last_hidden_state"] if isinstance(outputs, dict) else outputs.encoder_last_hidden_state
            
            inner = getattr(model, "model", model)
            dec_layer = inner.model.decoder.layers[layer_idx]
            dec_state = dec_states[layer_idx]
            
            D = dec_state.size(-1)
            H = inner.config.decoder_attention_heads
            d_k = D // H
            
            q = dec_layer.encoder_attn.q_proj(dec_state).view(1, -1, H, d_k).transpose(1, 2)
            k = dec_layer.encoder_attn.k_proj(enc_states).view(1, -1, H, d_k).transpose(1, 2)
            scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
            attn_map = F.softmax(scores, dim=-1)[0].mean(dim=0).cpu().numpy()

    # Tính toán chỉ số Attention Sink & Entropy
    T, S = attn_map.shape
    sink_mass = float(np.mean(attn_map[:, 0]) * 100.0) # Tỷ lệ % dồn vào token <s>
    eps = 1e-8
    entropy = float(-np.mean(np.sum(attn_map * np.log(attn_map + eps), axis=-1)))

    return attn_map, src_labels, tgt_labels, sink_mass, entropy

def plot_comparison(attn_v, src_v, tgt_v, sink_v, ent_v,
                    attn_t, src_t, tgt_t, sink_t, ent_t,
                    lang_name, output_png, output_pdf):
    """
    Renders side-by-side heatmaps formatted for ACL/EMNLP camera-ready standards.
    """
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300)
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["axes.linewidth"] = 0.8

    cmap = "Blues"

    # --- Subplot (a): Vanilla BARTpho ---
    ax1 = axes[0]
    if HAS_SEABORN:
        sns.heatmap(attn_v, ax=ax1, cmap=cmap, cbar=True, cbar_kws={"shrink": 0.75},
                    xticklabels=src_v, yticklabels=tgt_v, vmin=0.0, vmax=0.8)
    else:
        im1 = ax1.imshow(attn_v, cmap=cmap, vmin=0.0, vmax=0.8, aspect="auto")
        fig.colorbar(im1, ax=ax1, shrink=0.75)
        ax1.set_xticks(range(len(src_v)))
        ax1.set_yticks(range(len(tgt_v)))
        ax1.set_xticklabels(src_v)
        ax1.set_yticklabels(tgt_v)

    ax1.set_title(f"(a) Vanilla BARTpho (Baseline)\nAttention Sink on <s>: {sink_v:.1f}% | Entropy H(α): {ent_v:.3f}",
                  fontsize=13, fontweight="bold", pad=12, color="#990000")
    ax1.set_xlabel(f"Source Subwords ({lang_name})", fontsize=11, fontweight="bold", labelpad=8)
    ax1.set_ylabel("Target Subwords (Vietnamese)", fontsize=11, fontweight="bold", labelpad=8)
    ax1.tick_params(axis="x", rotation=45, labelsize=9)
    ax1.tick_params(axis="y", rotation=0, labelsize=9)

    # --- Subplot (b): TSSA (Ours) ---
    ax2 = axes[1]
    if HAS_SEABORN:
        sns.heatmap(attn_t, ax=ax2, cmap=cmap, cbar=True, cbar_kws={"shrink": 0.75},
                    xticklabels=src_t, yticklabels=tgt_t, vmin=0.0, vmax=0.8)
    else:
        im2 = ax2.imshow(attn_t, cmap=cmap, vmin=0.0, vmax=0.8, aspect="auto")
        fig.colorbar(im2, ax=ax2, shrink=0.75)
        ax2.set_xticks(range(len(src_t)))
        ax2.set_yticks(range(len(tgt_t)))
        ax2.set_xticklabels(src_t)
        ax2.set_yticklabels(tgt_t)

    reduction = ((sink_v - sink_t) / max(1e-5, sink_v)) * 100.0
    ax2.set_title(f"(b) TSSA (Ours)\nAttention Sink on <s>: {sink_t:.1f}% (↓ {reduction:.1f}%) | Entropy H(α): {ent_t:.3f}",
                  fontsize=13, fontweight="bold", pad=12, color="#006600")
    ax2.set_xlabel(f"Source Subwords ({lang_name})", fontsize=11, fontweight="bold", labelpad=8)
    ax2.set_ylabel("Target Subwords (Vietnamese)", fontsize=11, fontweight="bold", labelpad=8)
    ax2.tick_params(axis="x", rotation=45, labelsize=9)
    ax2.tick_params(axis="y", rotation=0, labelsize=9)

    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.savefig(output_pdf, bbox_inches="tight")
    plt.close()
    print(f"[+] Đã lưu thành công biểu đồ Heatmap:")
    print(f"    - PNG (Hình ảnh độ phân giải cao): {output_png}")
    print(f"    - PDF (Định dạng vector cho LaTeX): {output_pdf}")

def find_model_path(ckpt_dir, fallback_pretrained="vinai/bartpho-syllable"):
    if not ckpt_dir or not os.path.exists(ckpt_dir):
        print(f"[!] Thư mục {ckpt_dir} không tồn tại. Fallback về: {fallback_pretrained}")
        return fallback_pretrained

    files = os.listdir(ckpt_dir)
    
    # 1. Có file trọng số trực tiếp trong thư mục gốc?
    if any(f.endswith((".safetensors", ".bin", ".pt")) for f in files):
        print(f"[*] Tìm thấy file trọng số trong thư mục gốc: {ckpt_dir}")
        return ckpt_dir

    # 2. Có các sub-checkpoint (như trường hợp của bartpho_vanilla)?
    sub_ckpts = [os.path.join(ckpt_dir, d) for d in files if d.startswith("checkpoint-") and os.path.isdir(os.path.join(ckpt_dir, d))]
    if sub_ckpts:
        # Kiểm tra xem có trainer_state.json ghi nhận best_model_checkpoint không
        best_cand = None
        ts_path = os.path.join(ckpt_dir, "trainer_state.json")
        if os.path.exists(ts_path):
            try:
                data = json.load(open(ts_path))
                b_path = data.get("best_model_checkpoint")
                if b_path and os.path.exists(b_path):
                    best_cand = b_path
            except Exception:
                pass

        if not best_cand:
            for sc in sub_ckpts:
                sc_ts = os.path.join(sc, "trainer_state.json")
                if os.path.exists(sc_ts):
                    try:
                        data = json.load(open(sc_ts))
                        b_path = data.get("best_model_checkpoint")
                        if b_path and os.path.exists(b_path):
                            best_cand = b_path
                            break
                    except Exception:
                        pass

        if best_cand and os.path.exists(best_cand):
            print(f"[*] [Best Model] Đã chọn chuẩn xác checkpoint tốt nhất theo Trainer: {best_cand}")
            return best_cand

        # Nếu không có ghi nhận best, sắp xếp theo step cao nhất (checkpoint hội tụ cuối)
        def get_step(sc_path):
            bn = os.path.basename(sc_path)
            try:
                return int(bn.split("-")[-1])
            except Exception:
                return 0

        sub_ckpts.sort(key=get_step, reverse=True)
        chosen = sub_ckpts[0]
        print(f"[*] Đã tự động chọn sub-checkpoint có step cao nhất: {chosen}")
        return chosen

    # 3. Fallback nếu không tìm thấy file trọng số
    print(f"[!] Thư mục {ckpt_dir} chỉ có file kết quả ({files}), không có trọng số. Fallback: {fallback_pretrained}")
    return fallback_pretrained

def generate_heatmap_for_lang(lang, args, tokenizer):
    v_target = args.vanilla_ckpt or f"checkpoints/bartpho_vanilla_{lang}"
    t_target = args.tssa_ckpt or f"checkpoints/tssa_{lang}"

    v_ckpt = find_model_path(v_target, fallback_pretrained="vinai/bartpho-syllable")
    t_ckpt = find_model_path(t_target, fallback_pretrained="vinai/bartpho-syllable")

    # Nạp dữ liệu test
    test_csv = os.path.join(args.data_dir, lang, "test.csv")
    if not os.path.exists(test_csv):
        alt_csv = os.path.join("data", lang, "test.csv")
        if os.path.exists(alt_csv):
            test_csv = alt_csv
        else:
            print(f"[!] Không tìm thấy file test.csv tại: {test_csv}")
            return

    df = pd.read_csv(test_csv)

    # Chọn câu phù hợp (8 đến 18 từ để ma trận hiển thị đẹp, rõ ràng nhất)
    if args.sample_idx is not None:
        idx = args.sample_idx
    else:
        candidates = []
        for i, row in df.iterrows():
            s_len = len(str(row["src_text"]).split())
            t_len = len(str(row["tgt_text"]).split())
            if 8 <= s_len <= 16 and 8 <= t_len <= 16:
                candidates.append(i)
        idx = candidates[0] if len(candidates) > 0 else 0

    sample_row = df.iloc[idx]
    src_text = str(sample_row["src_text"])
    tgt_text = str(sample_row["tgt_text"])

    print("=" * 80)
    print(f"[*] Trực quan hóa Attention Heatmap cho: {lang.upper()} (Mẫu #{idx})")
    print(f"    - Nguồn ({lang}): {src_text}")
    print(f"    - Đích (Việt):   {tgt_text}")
    print("=" * 80)

    # 1. Nạp và trích xuất từ Vanilla BARTpho
    print(f"[*] [{lang.upper()}] Đang tính toán Cross-Attention cho Vanilla BARTpho từ: {v_ckpt}...")
    try:
        model_v = AutoModelForSeq2SeqLM.from_pretrained(v_ckpt).to(args.device)
    except Exception:
        try:
            model_v = TSSASeq2SeqModel(model_name_or_path=v_ckpt).to(args.device)
        except Exception:
            model_v = AutoModelForSeq2SeqLM.from_pretrained("vinai/bartpho-syllable").to(args.device)
    attn_v, src_v, tgt_v, sink_v, ent_v = extract_cross_attention(model_v, tokenizer, src_text, tgt_text, device=args.device)
    del model_v
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. Nạp và trích xuất từ TSSA
    print(f"[*] [{lang.upper()}] Đang tính toán Cross-Attention cho TSSA (Ours) từ: {t_ckpt}...")
    try:
        model_t = TSSASeq2SeqModel(model_name_or_path=t_ckpt).to(args.device)
    except Exception:
        model_t = AutoModelForSeq2SeqLM.from_pretrained(t_ckpt).to(args.device)
    attn_t, src_t, tgt_t, sink_t, ent_t = extract_cross_attention(model_t, tokenizer, src_text, tgt_text, device=args.device)
    del model_t
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    lang_display = {"rhade": "Rhade / Ê Đê", "tay": "Tay / Tày", "bahnaric": "Bahnaric / Ba Na"}.get(lang, lang)
    out_png = os.path.join(args.output_dir, f"attention_heatmap_{lang}.png")
    out_pdf = os.path.join(args.output_dir, f"attention_heatmap_{lang}.pdf")

    plot_comparison(
        attn_v, src_v, tgt_v, sink_v, ent_v,
        attn_t, src_t, tgt_t, sink_t, ent_t,
        lang_name=lang_display,
        output_png=out_png,
        output_pdf=out_pdf
    )

def main():
    parser = argparse.ArgumentParser(description="Cross-Attention Heatmap Generator (TSSA vs. Vanilla)")
    parser.add_argument("--lang", type=str, default="all", choices=["all", "rhade", "tay", "bahnaric"],
                        help="Ngôn ngữ cần trực quan hóa (mặc định: all để vẽ cả 3 tiếng)")
    parser.add_argument("--vanilla_ckpt", type=str, default=None, help="Đường dẫn checkpoint Vanilla BARTpho")
    parser.add_argument("--tssa_ckpt", type=str, default=None, help="Đường dẫn checkpoint TSSA")
    parser.add_argument("--data_dir", type=str, default="data_processed", help="Thư mục dữ liệu")
    parser.add_argument("--sample_idx", type=int, default=None, help="Chỉ mục câu cần vẽ (tự động chọn nếu None)")
    parser.add_argument("--output_dir", type=str, default="docs/figures", help="Thư mục lưu hình")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-syllable")
    langs = ["rhade", "tay", "bahnaric"] if args.lang == "all" else [args.lang]

    for lang in langs:
        generate_heatmap_for_lang(lang, args, tokenizer)

if __name__ == "__main__":
    main()
