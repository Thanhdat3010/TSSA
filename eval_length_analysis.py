"""
Length-Bucket & Hard-Instance Performance Analyzer (EMNLP/ACL Standard)
Slices test set predictions by:
1. Sentence Length Buckets: Short (<=12 words), Medium (13-25 words), Long (>25 words)
2. Hard vs. Easy Instance Slices (Lowest Vanilla performance vs. Rest, inspired by DPO-Align EMNLP 2024)

Ultra-fast execution:
- BLEU & chrF++ finish in < 1 second!
- If --comet is passed, sentence scores are computed ONCE on GPU and sliced in memory (no repeated calls).
"""

import os
import argparse
import numpy as np
import pandas as pd
import sacrebleu
import torch

COMET_MODEL = None

def get_comet_model():
    global COMET_MODEL
    if COMET_MODEL is None:
        try:
            from comet import download_model, load_from_checkpoint
            print("[*] Đang nạp mô hình COMET (Unbabel/wmt20-comet-da)...")
            model_path = download_model("Unbabel/wmt20-comet-da")
            COMET_MODEL = load_from_checkpoint(model_path)
            if torch.cuda.is_available():
                COMET_MODEL = COMET_MODEL.to("cuda")
        except Exception as e:
            print(f"[!] Không thể nạp mô hình COMET: {e}")
            COMET_MODEL = False
    return COMET_MODEL if COMET_MODEL is not False else None

def compute_all_comet_scores(srcs, preds, refs):
    """Tính toàn bộ sentence-level COMET một lần duy nhất trên GPU (khoảng 3-5 giây trên A100)."""
    c_mod = get_comet_model()
    if not c_mod:
        return None
    data = [{"src": s, "mt": p, "ref": r} for s, p, r in zip(srcs, preds, refs)]
    use_gpu = 1 if torch.cuda.is_available() else 0
    try:
        out = c_mod.predict(
            data,
            batch_size=64,
            gpus=use_gpu,
            accelerator="gpu" if use_gpu else "cpu",
            progress_bar=True
        )
        return out.scores
    except Exception as e:
        print(f"[!] Lỗi khi chạy COMET trên GPU: {e}, fallback CPU...")
        out = c_mod.predict(data, batch_size=32, gpus=0)
        return out.scores

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

def analyze_language_length(lang, checkpoints_dir="checkpoints", run_comet=False, short_thresh=12, long_thresh=25):
    vanilla_path = os.path.join(checkpoints_dir, f"bartpho_vanilla_{lang}", "test_predictions.csv")
    tssa_path = os.path.join(checkpoints_dir, f"tssa_{lang}", "test_predictions.csv")
    
    df_v = load_predictions(vanilla_path)
    df_t = load_predictions(tssa_path)
    
    if df_v is None or df_t is None:
        print(f"[!] Bỏ qua {lang.upper()}: Không tìm thấy đủ file dự đoán (Vanilla: {os.path.exists(vanilla_path)}, TSSA: {os.path.exists(tssa_path)})")
        return None, None
        
    refs = df_v["clean_ref"].tolist()
    srcs = df_v["clean_src"].tolist()
    v_preds = df_v["clean_pred"].tolist()
    t_preds = df_t["clean_pred"].tolist()
    lengths = [len(r.split()) for r in refs]

    # Tính COMET 1 lần duy nhất cho toàn bộ test set nếu được bật
    comet_scores_v = None
    comet_scores_t = None
    if run_comet and srcs:
        print(f"[*] [{lang.upper()}] Tính điểm COMET toàn bộ tập test trên GPU cho Vanilla...")
        comet_scores_v = compute_all_comet_scores(srcs, v_preds, refs)
        print(f"[*] [{lang.upper()}] Tính điểm COMET toàn bộ tập test trên GPU cho TSSA...")
        comet_scores_t = compute_all_comet_scores(srcs, t_preds, refs)

    buckets = [
        ("Short (<= 12 words)", lambda l: l <= short_thresh),
        ("Medium (13 - 25 words)", lambda l: (l > short_thresh) & (l <= long_thresh)),
        ("Long (> 25 words)", lambda l: l > long_thresh),
        ("All Instances", lambda l: True)
    ]
    
    length_rows = []
    for b_name, condition in buckets:
        idx_sub = [i for i, l in enumerate(lengths) if condition(l)]
        n_samples = len(idx_sub)
        if n_samples == 0:
            continue
            
        sub_refs = [refs[i] for i in idx_sub]
        sub_v_preds = [v_preds[i] for i in idx_sub]
        sub_t_preds = [t_preds[i] for i in idx_sub]
        
        # SacreBLEU
        b_v = round(sacrebleu.corpus_bleu(sub_v_preds, [sub_refs], smooth_method="exp").score, 2)
        b_t = round(sacrebleu.corpus_bleu(sub_t_preds, [sub_refs], smooth_method="exp").score, 2)
        delta_b = round(b_t - b_v, 2)
        
        # chrF++
        c_v = round(sacrebleu.corpus_chrf(sub_v_preds, [sub_refs], word_order=2).score, 2)
        c_t = round(sacrebleu.corpus_chrf(sub_t_preds, [sub_refs], word_order=2).score, 2)
        delta_c = round(c_t - c_v, 2)
        
        # COMET từ scores đã cache trong RAM
        if comet_scores_v and comet_scores_t:
            comet_v = round(float(np.mean([comet_scores_v[i] for i in idx_sub])), 4)
            comet_t = round(float(np.mean([comet_scores_t[i] for i in idx_sub])), 4)
            delta_comet = f"{comet_t - comet_v:+.4f}"
        else:
            comet_v, comet_t, delta_comet = "--", "--", "--"
        
        row = {
            "Ngôn Ngữ": lang.upper(),
            "Nhóm Độ Dài": b_name,
            "Số Mẫu (N)": n_samples,
            "Vanilla BLEU": b_v,
            "TSSA BLEU": b_t,
            "Δ BLEU": f"{delta_b:+.2f}",
            "Vanilla chrF++": c_v,
            "TSSA chrF++": c_t,
            "Δ chrF++": f"{delta_c:+.2f}",
        }
        if run_comet:
            row["Vanilla COMET"] = comet_v
            row["TSSA COMET"] = comet_t
            row["Δ COMET"] = delta_comet
        length_rows.append(row)
        
    # --- HARD INSTANCE SLICING (EMNLP 2024 Style) ---
    sent_scores = [sacrebleu.sentence_chrf(p, [r], word_order=2).score for p, r in zip(v_preds, refs)]
    q25 = np.percentile(sent_scores, 25)
    hard_idx = [i for i, s in enumerate(sent_scores) if s <= q25]
    easy_idx = [i for i, s in enumerate(sent_scores) if s > q25]
    
    hard_rows = []
    for subset_name, indices in [("Hard Instances (Bottom 25%)", hard_idx), ("Easy Instances (Top 75%)", easy_idx)]:
        sub_refs = [refs[i] for i in indices]
        sub_v_preds = [v_preds[i] for i in indices]
        sub_t_preds = [t_preds[i] for i in indices]
        
        b_v = round(sacrebleu.corpus_bleu(sub_v_preds, [sub_refs], smooth_method="exp").score, 2)
        b_t = round(sacrebleu.corpus_bleu(sub_t_preds, [sub_refs], smooth_method="exp").score, 2)
        delta_b = round(b_t - b_v, 2)
        
        c_v = round(sacrebleu.corpus_chrf(sub_v_preds, [sub_refs], word_order=2).score, 2)
        c_t = round(sacrebleu.corpus_chrf(sub_t_preds, [sub_refs], word_order=2).score, 2)
        delta_c = round(c_t - c_v, 2)
        
        if comet_scores_v and comet_scores_t:
            comet_v = round(float(np.mean([comet_scores_v[i] for i in indices])), 4)
            comet_t = round(float(np.mean([comet_scores_t[i] for i in indices])), 4)
            delta_comet = f"{comet_t - comet_v:+.4f}"
        else:
            comet_v, comet_t, delta_comet = "--", "--", "--"
        
        h_row = {
            "Ngôn Ngữ": lang.upper(),
            "Phân Hạng Độ Khó": subset_name,
            "Số Mẫu (N)": len(indices),
            "Vanilla BLEU": b_v,
            "TSSA BLEU": b_t,
            "Δ BLEU": f"{delta_b:+.2f}",
            "Vanilla chrF++": c_v,
            "TSSA chrF++": c_t,
            "Δ chrF++": f"{delta_c:+.2f}",
        }
        if run_comet:
            h_row["Vanilla COMET"] = comet_v
            h_row["TSSA COMET"] = comet_t
            h_row["Δ COMET"] = delta_comet
        hard_rows.append(h_row)
        
    return length_rows, hard_rows

def main():
    parser = argparse.ArgumentParser(description="TSSA Length & Hard Instance Performance Slicing")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints", help="Thư mục chứa checkpoints")
    parser.add_argument("--lang", type=str, default="all", choices=["all", "rhade", "tay", "bahnaric"])
    parser.add_argument("--comet", action="store_true", help="Bật tính thêm COMET trên GPU (nhanh)")
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
    
    print("\n--- 1. HIỆU NĂNG PHÂN THEO ĐỘ DÀI CÂU (SENTENCE LENGTH BUCKETS) ---")
    print(df_length.to_markdown(index=False))
    
    print("\n--- 2. HIỆU NĂNG PHÂN THEO CÂU KHÓ VS. CÂU DỄ (HARD VS. EASY SLICING - EMNLP STYLE) ---")
    print(df_hard.to_markdown(index=False))
    print("=" * 110)
    
    os.makedirs(os.path.dirname(args.output_md), exist_ok=True)
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write("# 🔬 Báo Cáo Phân Tích Độ Bền Vững: Độ Dài Câu & Câu Khó (Length & Hard Instances Analysis)\n\n")
        f.write("Báo cáo này thẩm định độ bền bỉ của **TSSA** so với **Vanilla BARTpho** khi câu dài dần và khi đối mặt với các trường hợp ngữ nghĩa khó (Hard Instances).\n\n")
        f.write("## 1. Hiệu Năng Phân Bổ Theo Độ Dài Câu (Sentence Length Buckets)\n\n")
        f.write(df_length.to_markdown(index=False))
        f.write("\n\n## 2. Hiệu Năng Trên Câu Khó vs. Câu Dễ (Hard vs. Easy Instances)\n\n")
        f.write(df_hard.to_markdown(index=False))
        f.write("\n\n---\n*Báo cáo được tự động sinh bởi `eval_length_analysis.py`.*\n")
        
    print(f"[+] Đã lưu báo cáo chi tiết vào: {args.output_md}")

if __name__ == "__main__":
    main()
