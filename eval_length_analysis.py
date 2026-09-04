"""
Length-Bucket & Hard-Instance Performance Analyzer (EMNLP/ACL Standard)
Slices test set predictions by:
1. Sentence Length Buckets: Short (<=12 words), Medium (13-25 words), Long (>25 words)
2. Hard vs. Easy Instance Slices (Lowest Vanilla performance vs. Rest, inspired by DPO-Align EMNLP 2024)

Computes SacreBLEU, chrF++, and optional COMET to evaluate robustness against Attention Sinking.
"""

import os
import argparse
import numpy as np
import pandas as pd
import sacrebleu

# Optional COMET loader
COMET_MODEL = None
def get_comet_model():
    global COMET_MODEL
    if COMET_MODEL is None:
        try:
            from comet import download_model, load_from_checkpoint
            print("[*] Đang nạp mô hình COMET (Unbabel/wmt20-comet-da)...")
            model_path = download_model("Unbabel/wmt20-comet-da")
            COMET_MODEL = load_from_checkpoint(model_path)
        except Exception as e:
            print(f"[!] Không thể nạp mô hình COMET: {e}")
            COMET_MODEL = False
    return COMET_MODEL if COMET_MODEL is not False else None

def load_predictions(pred_path):
    if not os.path.exists(pred_path):
        return None
    df = pd.read_csv(pred_path)
    src_col, ref_col, pred_col = None, None, None
    for col in df.columns:
        cl = col.lower()
        if "src" in cl or "source" in cl:
            src_col = col
        elif "ref" in cl or "target" in cl:
            ref_col = col
        elif "pred" in cl or "translation" in cl or "hyp" in cl:
            pred_col = col
    
    if not (ref_col and pred_col):
        return None
        
    df["clean_ref"] = df[ref_col].astype(str).str.strip()
    df["clean_pred"] = df[pred_col].astype(str).str.strip()
    df["clean_src"] = df[src_col].astype(str).str.strip() if src_col else ""
    return df

def compute_corpus_metrics(preds, refs, srcs=None, run_comet=False):
    if len(preds) == 0 or len(refs) == 0:
        return 0.0, 0.0, "--"
    b_res = sacrebleu.corpus_bleu(preds, [refs], smooth_method="exp")
    c_res = sacrebleu.corpus_chrf(preds, [refs], word_order=2)
    b_score = round(b_res.score, 2)
    c_score = round(c_res.score, 2)
    
    comet_score = "--"
    if run_comet and srcs is not None:
        c_mod = get_comet_model()
        if c_mod:
            try:
                data = [{"src": s, "mt": p, "ref": r} for s, p, r in zip(srcs, preds, refs)]
                out = c_mod.predict(data, batch_size=32, gpus=1 if c_mod.device.type == "cuda" else 0)
                comet_score = round(float(out.system_score), 4)
            except Exception:
                pass
    return b_score, c_score, comet_score

def analyze_language_length(lang, checkpoints_dir="checkpoints", run_comet=False, short_thresh=12, long_thresh=25):
    vanilla_path = os.path.join(checkpoints_dir, f"bartpho_vanilla_{lang}", "test_predictions.csv")
    tssa_path = os.path.join(checkpoints_dir, f"tssa_{lang}", "test_predictions.csv")
    
    df_v = load_predictions(vanilla_path)
    df_t = load_predictions(tssa_path)
    
    if df_v is None or df_t is None:
        print(f"[!] Thiếu file dự đoán cho {lang} (Vanilla: {os.path.exists(vanilla_path)}, TSSA: {os.path.exists(tssa_path)})")
        return None, None
        
    # Tính word count dựa trên target reference
    lengths = [len(r.split()) for r in df_v["clean_ref"]]
    df_v["len"] = lengths
    df_t["len"] = lengths
    
    buckets = [
        ("Short (<= 12 words)", lambda l: l <= short_thresh),
        ("Medium (13 - 25 words)", lambda l: (l > short_thresh) & (l <= long_thresh)),
        ("Long (> 25 words)", lambda l: l > long_thresh),
        ("All Instances", lambda l: True)
    ]
    
    length_rows = []
    for b_name, condition in buckets:
        mask = [condition(l) for l in lengths]
        idx_sub = [i for i, m in enumerate(mask) if m]
        n_samples = len(idx_sub)
        
        if n_samples == 0:
            continue
            
        sub_refs = [df_v["clean_ref"].iloc[i] for i in idx_sub]
        sub_srcs = [df_v["clean_src"].iloc[i] for i in idx_sub]
        v_preds = [df_v["clean_pred"].iloc[i] for i in idx_sub]
        t_preds = [df_t["clean_pred"].iloc[i] for i in idx_sub]
        
        b_v, c_v, comet_v = compute_corpus_metrics(v_preds, sub_refs, sub_srcs, run_comet=run_comet)
        b_t, c_t, comet_t = compute_corpus_metrics(t_preds, sub_refs, sub_srcs, run_comet=run_comet)
        
        delta_b = round(b_t - b_v, 2)
        delta_c = round(c_t - c_v, 2)
        
        length_rows.append({
            "Ngôn Ngữ": lang.upper(),
            "Nhóm Độ Dài": b_name,
            "Số Mẫu (N)": n_samples,
            "Vanilla BLEU": b_v,
            "TSSA BLEU": b_t,
            "Δ BLEU": f"{delta_b:+.2f}",
            "Vanilla chrF++": c_v,
            "TSSA chrF++": c_t,
            "Δ chrF++": f"{delta_c:+.2f}",
            "Vanilla COMET": comet_v,
            "TSSA COMET": comet_t
        })
        
    # --- HARD INSTANCE SLICING (EMNLP 2024 Style) ---
    # Tính sentence-level chrF++ cho Vanilla để phân hạng độ khó
    sent_scores = []
    for p, r in zip(df_v["clean_pred"], df_v["clean_ref"]):
        score = sacrebleu.sentence_chrf(p, [r], word_order=2).score
        sent_scores.append(score)
    df_v["difficulty_score"] = sent_scores
    
    # 25% câu điểm thấp nhất = Hard, 75% còn lại = Easy
    q25 = np.percentile(sent_scores, 25)
    hard_idx = [i for i, s in enumerate(sent_scores) if s <= q25]
    easy_idx = [i for i, s in enumerate(sent_scores) if s > q25]
    
    hard_rows = []
    for subset_name, indices in [("Hard Instances (Bottom 25%)", hard_idx), ("Easy Instances (Top 75%)", easy_idx)]:
        sub_refs = [df_v["clean_ref"].iloc[i] for i in indices]
        sub_srcs = [df_v["clean_src"].iloc[i] for i in indices]
        v_preds = [df_v["clean_pred"].iloc[i] for i in indices]
        t_preds = [df_t["clean_pred"].iloc[i] for i in indices]
        
        b_v, c_v, comet_v = compute_corpus_metrics(v_preds, sub_refs, sub_srcs, run_comet=run_comet)
        b_t, c_t, comet_t = compute_corpus_metrics(t_preds, sub_refs, sub_srcs, run_comet=run_comet)
        
        delta_b = round(b_t - b_v, 2)
        delta_c = round(c_t - c_v, 2)
        
        hard_rows.append({
            "Ngôn Ngữ": lang.upper(),
            "Phân Hạng Độ Khó": subset_name,
            "Số Mẫu (N)": len(indices),
            "Vanilla BLEU": b_v,
            "TSSA BLEU": b_t,
            "Δ BLEU": f"{delta_b:+.2f}",
            "Vanilla chrF++": c_v,
            "TSSA chrF++": c_t,
            "Δ chrF++": f"{delta_c:+.2f}",
            "Vanilla COMET": comet_v,
            "TSSA COMET": comet_t
        })
        
    return length_rows, hard_rows

def main():
    parser = argparse.ArgumentParser(description="TSSA Length & Hard Instance Performance Slicing")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints", help="Thư mục chứa checkpoints")
    parser.add_argument("--lang", type=str, default="all", choices=["all", "rhade", "tay", "bahnaric"])
    parser.add_argument("--comet", action="store_true", help="Tính thêm điểm COMET cho từng nhóm")
    parser.add_argument("--short_thresh", type=int, default=12, help="Ngưỡng câu ngắn (<= N từ)")
    parser.add_argument("--long_thresh", type=int, default=25, help="Ngưỡng câu dài (> N từ)")
    parser.add_argument("--output_md", type=str, default="docs/LENGTH_AND_HARD_ANALYSIS.md", help="Đường dẫn lưu báo cáo")
    args = parser.parse_args()

    langs = ["rhade", "tay", "bahnaric"] if args.lang == "all" else [args.lang]
    
    all_length_data = []
    all_hard_data = []
    
    print("=" * 110)
    print("      📊 BÁO CÁO BÓC TÁCH HIỆU NĂNG THEO ĐỘ DÀI CÂU & CÂU KHÓ (LENGTH & HARD-INSTANCE SLICING)")
    print("=" * 110)

    for lang in langs:
        print(f"[*] Đang phân tích ngôn ngữ: {lang.upper()}...")
        l_rows, h_rows = analyze_language_length(
            lang,
            checkpoints_dir=args.checkpoints_dir,
            run_comet=args.comet,
            short_thresh=args.short_thresh,
            long_thresh=args.long_thresh
        )
        if l_rows:
            all_length_data.extend(l_rows)
        if h_rows:
            all_hard_data.extend(h_rows)

    if not all_length_data:
        print("[!] Không tìm thấy dữ liệu dự đoán hợp lệ để bóc tách.")
        return

    df_length = pd.DataFrame(all_length_data)
    df_hard = pd.DataFrame(all_hard_data)
    
    # In ra Terminal
    print("\n--- 1. HIỆU NĂNG PHÂN THEO ĐỘ DÀI CÂU (SENTENCE LENGTH BUCKETS) ---")
    cols_len = ["Ngôn Ngữ", "Nhóm Độ Dài", "Số Mẫu (N)", "Vanilla BLEU", "TSSA BLEU", "Δ BLEU", "Vanilla chrF++", "TSSA chrF++", "Δ chrF++"]
    if args.comet:
        cols_len.extend(["Vanilla COMET", "TSSA COMET"])
    print(df_length[cols_len].to_markdown(index=False))
    
    print("\n--- 2. HIỆU NĂNG PHÂN THEO CÂU KHÓ VS. CÂU DỄ (HARD VS. EASY SLICING - EMNLP STYLE) ---")
    cols_hard = ["Ngôn Ngữ", "Phân Hạng Độ Khó", "Số Mẫu (N)", "Vanilla BLEU", "TSSA BLEU", "Δ BLEU", "Vanilla chrF++", "TSSA chrF++", "Δ chrF++"]
    if args.comet:
        cols_hard.extend(["Vanilla COMET", "TSSA COMET"])
    print(df_hard[cols_hard].to_markdown(index=False))
    print("=" * 110)
    
    # Lưu file Markdown báo cáo
    os.makedirs(os.path.dirname(args.output_md), exist_ok=True)
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write("# 🔬 Báo Cáo Phân Tích Độ Bền Vững: Độ Dài Câu & Câu Khó (Length & Hard Instances Analysis)\n\n")
        f.write("Báo cáo này thẩm định độ bền bỉ của **TSSA** so với **Vanilla BARTpho** khi câu dài dần và khi đối mặt với các trường hợp ngữ nghĩa khó (Hard Instances).\n\n")
        f.write("## 1. Hiệu Năng Phân Bổ Theo Độ Dài Câu (Sentence Length Buckets)\n\n")
        f.write(df_length[cols_len].to_markdown(index=False))
        f.write("\n\n## 2. Hiệu Năng Trên Câu Khó vs. Câu Dễ (Hard vs. Easy Instances)\n\n")
        f.write(df_hard[cols_hard].to_markdown(index=False))
        f.write("\n\n---\n*Báo cáo được tự động sinh bởi `eval_length_analysis.py`.*\n")
        
    print(f"[+] Đã lưu báo cáo chi tiết vào: {args.output_md}")

if __name__ == "__main__":
    main()
