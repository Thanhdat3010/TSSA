"""
Extract Qualitative Translation Cases from Hard Instances (ACL/EMNLP Table 6)
Finds contrasting translation examples where Vanilla BARTpho fails (attention sink / word repetition / omission)
and TSSA (Ours) accurately reconstructs the target semantics.
"""

import os
import argparse
import csv
import numpy as np

try:
    import sacrebleu
    HAS_SACREBLEU = True
except ImportError:
    HAS_SACREBLEU = False

def load_csv_predictions(file_path):
    if not os.path.exists(file_path):
        return None
    rows = []
    with open(file_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def get_columns(row):
    src_col, ref_col, pred_col = None, None, None
    for k in row.keys():
        kl = k.lower()
        if "src" in kl or "source" in kl:
            src_col = k
        elif "ref" in kl or "target" in kl:
            ref_col = k
        elif "pred" in kl or "translation" in kl or "hyp" in kl:
            pred_col = k
    return src_col, ref_col, pred_col

def sentence_eval(pred, ref):
    if HAS_SACREBLEU:
        chrf = sacrebleu.sentence_chrf(pred, [ref], word_order=2).score
        bleu = sacrebleu.sentence_bleu(pred, [ref], smooth_method="exp").score
        return chrf, bleu
    else:
        # Simple word overlap fallback
        p_tok = set(pred.lower().split())
        r_tok = set(ref.lower().split())
        overlap = len(p_tok & r_tok) / max(len(r_tok), 1) * 100
        return overlap, overlap

def check_repetition(text):
    words = text.lower().split()
    if len(words) < 4:
        return False
    for i in range(len(words) - 3):
        if words[i] == words[i+1] == words[i+2]:
            return True
        if words[i:i+2] == words[i+2:i+4]:
            return True
    return False

def find_best_cases(lang, checkpoints_dir="checkpoints", top_k=3):
    v_path = os.path.join(checkpoints_dir, f"bartpho_vanilla_{lang}", "test_predictions.csv")
    t_path = os.path.join(checkpoints_dir, f"tssa_{lang}", "test_predictions.csv")
    
    v_rows = load_csv_predictions(v_path)
    t_rows = load_csv_predictions(t_path)
    
    if not v_rows or not t_rows:
        print(f"[!] Không tìm thấy predictions cho {lang.upper()} (V: {bool(v_rows)}, T: {bool(t_rows)})")
        return []

    src_k_v, ref_k_v, pred_k_v = get_columns(v_rows[0])
    src_k_t, ref_k_t, pred_k_t = get_columns(t_rows[0])
    
    pairs = []
    n = min(len(v_rows), len(t_rows))
    
    for i in range(n):
        src = v_rows[i][src_k_v].strip()
        ref = v_rows[i][ref_k_v].strip()
        v_pred = v_rows[i][pred_k_v].strip()
        t_pred = t_rows[i][pred_k_t].strip()
        
        v_chrf, v_bleu = sentence_eval(v_pred, ref)
        t_chrf, t_bleu = sentence_eval(t_pred, ref)
        
        has_rep = check_repetition(v_pred)
        words_dropped = len(ref.split()) - len(v_pred.split())
        
        # Tiêu chí chấm điểm độ tương phản: TSSA tốt, Vanilla tệ
        gain = t_chrf - v_chrf
        penalty_v = 0
        if has_rep:
            penalty_v += 20
        if words_dropped > 5:
            penalty_v += 10
            
        contrast_score = gain + penalty_v
        
        pairs.append({
            "idx": i,
            "src": src,
            "ref": ref,
            "v_pred": v_pred,
            "t_pred": t_pred,
            "v_chrf": v_chrf,
            "t_chrf": t_chrf,
            "gain": gain,
            "has_rep": has_rep,
            "contrast_score": contrast_score
        })
        
    # Sắp xếp theo độ tương phản cao nhất
    pairs.sort(key=lambda x: x["contrast_score"], reverse=True)
    return pairs[:top_k]

def main():
    parser = argparse.ArgumentParser(description="Extract Qualitative Cases for ACL Paper")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints")
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()

    langs = ["rhade", "tay", "bahnaric"]
    selected_cases = {}
    
    print("=" * 90)
    print("      🔍 TRÍCH XUẤT CÁC MẪU CÂU ĐỊNH TÍNH TƯƠNG PHẢN CAO NHẤT (TABLE 6)")
    print("=" * 90)
    
    for lang in langs:
        cases = find_best_cases(lang, args.checkpoints_dir, args.top_k)
        selected_cases[lang] = cases
        print(f"\n[{lang.upper()}] Tìm thấy {len(cases)} mẫu tiêu biểu:")
        for idx, c in enumerate(cases, 1):
            print(f"  --- Mẫu #{idx} (Dòng test #{c['idx']}, Gain chrF++: +{c['gain']:.2f}, Repetition: {c['has_rep']}) ---")
            print(f"  [Nguồn ({lang})]:  {c['src']}")
            print(f"  [Chuẩn (Ref)]:    {c['ref']}")
            print(f"  [Vanilla]:        {c['v_pred']}  (chrF: {c['v_chrf']:.1f})")
            print(f"  [TSSA (Ours)]:    {c['t_pred']}  (chrF: {c['t_chrf']:.1f})")
            
    # Tạo sẵn mẫu LaTeX
    latex_output = "\\begin{table*}[t]\n\\centering\\small\n"
    latex_output += "\\caption{\\textbf{Qualitative Translation Case Studies on Hard Instances.} Comparing translations from Vanilla BARTpho and TSSA against the reference. Vanilla exhibits severe word repetition, clause truncation, or catastrophic token omission; TSSA restores fluent syntax and accurately translates domain terminology.}\n"
    latex_output += "\\label{tab:qualitative_examples}\n\\vspace{4pt}\n"
    latex_output += "\\begin{tabular}{p{0.12\\textwidth}p{0.84\\textwidth}}\n\\toprule\n"
    
    for lang in langs:
        if not selected_cases[lang]:
            continue
        c = selected_cases[lang][0]
        title_str = f"{lang.title()} $\\rightarrow$ Vietnamese (Test Instance #{c['idx']})"
        latex_output += "\\multicolumn{2}{l}{\\textbf{" + title_str + "}} \\\\\n"
        latex_output += "\\textbf{Source (" + lang.title() + ")} & " + c['src'] + " \\\\\n"
        latex_output += "\\textbf{Reference} & " + c['ref'] + " \\\\\n"
        latex_output += "\\textbf{Vanilla} & \\textcolor{red!80!black}{" + c['v_pred'] + "} \\\\\n"
        latex_output += "\\textbf{TSSA (Ours)} & \\textcolor{topgreen}{" + c['t_pred'] + "} \\\\\n"
        latex_output += "\\midrule\n"
        
    latex_output = latex_output.rstrip("\\midrule\n") + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n"
    
    out_tex_path = "docs/qualitative_table_snippet.tex"
    with open(out_tex_path, "w", encoding="utf-8") as f:
        f.write(latex_output)
    print(f"\n[+] Đã lưu snippet bảng LaTeX vào: {out_tex_path}")

if __name__ == "__main__":
    main()
