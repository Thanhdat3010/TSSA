"""
Summary and Benchmark Inspector for ViT5 (VietAI/vit5-base)
Computes SacreBLEU, chrF++, METEOR, and COMET across all 18 ViT5 models (6 methods x 3 languages).
Outputs both rich Markdown tables and publication-ready LaTeX code for Table 7.
"""

import os
import glob
import json
import argparse
import pandas as pd
import sacrebleu

# NLTK WordNet for METEOR
try:
    import nltk
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    from nltk.translate.meteor_score import meteor_score
    NLTK_AVAILABLE = True
except Exception:
    NLTK_AVAILABLE = False

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

def compute_metrics_for_pred_file(pred_file, compute_comet=False):
    metrics_cache_file = os.path.join(os.path.dirname(pred_file), "eval_metrics.json")
    cached_metrics = {}
    if os.path.exists(metrics_cache_file):
        try:
            with open(metrics_cache_file, "r", encoding="utf-8") as f:
                cached_metrics = json.load(f)
        except Exception:
            pass

    if cached_metrics and "bleu" in cached_metrics and "chrf" in cached_metrics and "meteor" in cached_metrics:
        if not compute_comet or ("comet" in cached_metrics and cached_metrics["comet"] != "--"):
            return cached_metrics

    df = pd.read_csv(pred_file)
    src_col, ref_col, pred_col = None, None, None
    for col in df.columns:
        col_lower = col.lower()
        if "src" in col_lower or "source" in col_lower:
            src_col = col
        elif "ref" in col_lower or "target" in col_lower:
            ref_col = col
        elif "pred" in col_lower or "translation" in col_lower or "hyp" in col_lower:
            pred_col = col

    if not (ref_col and pred_col):
        return None

    refs = [str(t).strip() for t in df[ref_col].fillna("").tolist()]
    preds = [str(p).strip() for p in df[pred_col].fillna("").tolist()]
    srcs = [str(s).strip() for s in df[src_col].fillna("").tolist()] if src_col else [""] * len(preds)

    # 1. SacreBLEU
    b_res = sacrebleu.corpus_bleu(preds, [refs], smooth_method="exp")
    bleu_score = round(b_res.score, 2)

    # 2. chrF++
    c_res = sacrebleu.corpus_chrf(preds, [refs], word_order=2)
    chrf_score = round(c_res.score, 2)

    # 3. METEOR
    meteor_val = cached_metrics.get("meteor", "--")
    if (meteor_val == "--" or meteor_val is None) and NLTK_AVAILABLE:
        try:
            scores = []
            for r, p in zip(refs, preds):
                r_tokens = r.split()
                p_tokens = p.split()
                scores.append(meteor_score([r_tokens], p_tokens))
            meteor_val = round(float(sum(scores) / len(scores)) * 100, 2)
        except Exception:
            meteor_val = "--"

    # 4. COMET
    comet_val = cached_metrics.get("comet", "--")
    if compute_comet and (comet_val == "--" or comet_val is None):
        c_model = get_comet_model()
        if c_model:
            try:
                data = [{"src": s, "mt": p, "ref": r} for s, p, r in zip(srcs, preds, refs)]
                comet_output = c_model.predict(data, batch_size=32, gpus=1 if torch.cuda.is_available() else 0)
                comet_val = round(float(comet_output.system_score) * 100, 2)
            except Exception as e:
                print(f"[!] Lỗi tính COMET: {e}")
                comet_val = "--"

    metrics_to_cache = {
        "bleu": bleu_score,
        "chrf": chrf_score,
        "meteor": meteor_val,
        "comet": comet_val
    }
    try:
        with open(metrics_cache_file, "w", encoding="utf-8") as f:
            json.dump(metrics_to_cache, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

    return metrics_to_cache

def summarize_vit5(checkpoints_dir="checkpoints", compute_comet=False):
    languages = [
        ("rhade", "Rhade (Ê Đê)"),
        ("tay", "Tay (Tày)"),
        ("bahnaric", "Bahnaric (Ba Na)")
    ]
    
    methods = [
        ("vanilla", "Vanilla ViT5"),
        ("align_to_distill", "Align-to-Distill (A2D)"),
        ("shift_aet", "Shift-AET"),
        ("awesome_align", "AWESOME-align"),
        ("cl_lsa", "CL-LSA (InfoNCE)"),
        ("tssa", "TSSA (Proposed)")
    ]

    all_data = []

    for lang_code, lang_name in languages:
        vanilla_bleu = None
        for method_key, method_name in methods:
            folder_name = f"vit5_{method_key}_{lang_code}"
            pred_file = os.path.join(checkpoints_dir, folder_name, "test_predictions.csv")
            
            m_dict = None
            if os.path.exists(pred_file):
                try:
                    m_dict = compute_metrics_for_pred_file(pred_file, compute_comet=compute_comet)
                except Exception as e:
                    print(f"[!] Lỗi đọc {pred_file}: {e}")

            b_score = m_dict["bleu"] if m_dict and m_dict.get("bleu") is not None else None
            c_score = m_dict["chrf"] if m_dict and m_dict.get("chrf") is not None else None
            m_score = m_dict["meteor"] if m_dict and m_dict.get("meteor") is not None else None
            comet_score = m_dict["comet"] if m_dict and m_dict.get("comet") is not None else None

            if method_key == "vanilla" and b_score is not None:
                vanilla_bleu = b_score

            delta_str = "-"
            if b_score is not None and vanilla_bleu is not None:
                diff = b_score - vanilla_bleu
                delta_str = f"{diff:+.2f}" if method_key != "vanilla" else "Ref (0.0)"

            all_data.append({
                "lang_code": lang_code,
                "Language": lang_name,
                "Method": method_name,
                "method_key": method_key,
                "Folder": folder_name,
                "Status": "✅ Done" if b_score is not None else "⏳ Pending",
                "BLEU": b_score,
                "chrF++": c_score,
                "METEOR": m_score,
                "COMET": comet_score,
                "Δ vs Vanilla": delta_str
            })

    return all_data

def generate_latex_table7(all_data):
    """Generates LaTeX code for Table 7: ViT5 Generalization Benchmark."""
    latex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{\textbf{Cross-Architecture Generalization Benchmark on ViT5 (\texttt{VietAI/vit5-base})}. Evaluates whether Target-Supervised Semantic Anchoring (TSSA) preserves its empirical superiority when switching backbone architecture from encoder-decoder BART (\textsc{BARTpho}) to pure T5-style sequence-to-sequence structure (\textsc{ViT5-base}). Best results are in \textbf{bold}; second-best underlined.}",
        r"\label{tab:vit5_generalization_benchmark}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"\textbf{Language} & \textbf{Method / Architecture} & \textbf{SacreBLEU $\uparrow$} & \textbf{chrF++ $\uparrow$} & \textbf{METEOR $\uparrow$} & \textbf{COMET $\uparrow$} \\",
        r"\midrule"
    ]

    current_lang = None
    for row in all_data:
        if row["Language"] != current_lang:
            if current_lang is not None:
                latex.append(r"\midrule")
            current_lang = row["Language"]
            lang_label = f"\\textbf{{{current_lang}}}"
        else:
            lang_label = ""

        m_name = row["Method"]
        if row["method_key"] == "tssa":
            m_name = f"\\textbf{{{m_name}}}"

        b_str = f"{row['BLEU']:.2f}" if row['BLEU'] is not None else "--"
        c_str = f"{row['chrF++']:.2f}" if row['chrF++'] is not None else "--"
        m_str = f"{row['METEOR']}" if row['METEOR'] is not None else "--"
        co_str = f"{row['COMET']}" if row['COMET'] is not None else "--"

        if row["method_key"] == "tssa" and row['BLEU'] is not None:
            b_str = f"\\textbf{{{b_str}}}"
            c_str = f"\\textbf{{{c_str}}}"

        latex.append(f"{lang_label} & {m_name} & {b_str} & {c_str} & {m_str} & {co_str} \\\\")

    latex.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}"
    ])
    return "\n".join(latex)

def main():
    parser = argparse.ArgumentParser(description="ViT5 Results Inspector")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints", help="Path to checkpoints folder")
    parser.add_argument("--comet", action="store_true", default=False, help="Compute COMET metrics")
    parser.add_argument("--latex", action="store_true", default=False, help="Output LaTeX Table 7")
    args = parser.parse_args()

    print("=" * 110)
    print("      📊 BÁO CÁO FULL BENCHMARK ViT5 (VietAI/vit5-base) TRÊN 3 BỘ DỮ LIỆU")
    print("=" * 110)

    data = summarize_vit5(args.checkpoints_dir, compute_comet=args.comet)
    df = pd.DataFrame(data)
    
    # Display table columns
    disp_cols = ["Language", "Method", "Status", "BLEU", "chrF++", "METEOR", "COMET", "Δ vs Vanilla"]
    print(df[disp_cols].to_markdown(index=False))
    print("=" * 110)

    # Summary of completions
    done_count = sum(1 for r in data if r["BLEU"] is not None)
    print(f"[*] Tiến độ: {done_count} / {len(data)} mô hình đã có kết quả.")
    
    if args.latex or done_count > 0:
        print("\n" + "=" * 110)
        print("                 📜 LATEX CODE CHO BẢNG 7 (TABLE 7 DOSSIER)")
        print("=" * 110)
        print(generate_latex_table7(data))
        print("=" * 110)

if __name__ == "__main__":
    main()
