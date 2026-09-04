"""
Paired Bootstrap Resampling Significance Testing for ViT5 (VietAI/vit5-base)
Calculates p-values and 95% Confidence Intervals for:
1. vit5_tssa vs. vit5_vanilla (Architecture Baseline)
2. vit5_tssa vs. Best Competitor on ViT5 (A2D / Shift-AET / AWESOME / CL-LSA)
"""

import os
import argparse
import csv
import numpy as np
import sacrebleu

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

def load_predictions(file_path):
    if not os.path.exists(file_path):
        return None, None, None
    srcs, refs, preds = [], [], []
    with open(file_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        src_col, ref_col, pred_col = None, None, None
        for r in reader:
            if src_col is None:
                for k in r.keys():
                    kl = k.lower()
                    if "src" in kl or "source" in kl:
                        src_col = k
                    elif "ref" in kl or "target" in kl:
                        ref_col = k
                    elif "pred" in kl or "translation" in kl or "hyp" in kl:
                        pred_col = k
            srcs.append(r[src_col].strip() if src_col else "")
            refs.append(r[ref_col].strip())
            preds.append(r[pred_col].strip())
    return srcs, refs, preds

def paired_bootstrap_test(tssa_preds, comp_preds, refs, num_samples=1000, seed=42, desc=""):
    np.random.seed(seed)
    n = len(refs)
    
    real_bleu_tssa = sacrebleu.corpus_bleu(tssa_preds, [refs], smooth_method="exp").score
    real_bleu_comp = sacrebleu.corpus_bleu(comp_preds, [refs], smooth_method="exp").score
    real_delta_bleu = real_bleu_tssa - real_bleu_comp
    
    real_chrf_tssa = sacrebleu.corpus_chrf(tssa_preds, [refs], word_order=2).score
    real_chrf_comp = sacrebleu.corpus_chrf(comp_preds, [refs], word_order=2).score
    real_delta_chrf = real_chrf_tssa - real_chrf_comp

    delta_bleu_samples = []
    delta_chrf_samples = []

    pbar = tqdm(range(num_samples), desc=f"   {desc:<35}", unit="iter", ncols=100)
    for _ in pbar:
        idx = np.random.choice(n, size=n, replace=True)
        sub_refs = [refs[i] for i in idx]
        sub_tssa = [tssa_preds[i] for i in idx]
        sub_comp = [comp_preds[i] for i in idx]

        b_t = sacrebleu.corpus_bleu(sub_tssa, [sub_refs], smooth_method="exp").score
        b_c = sacrebleu.corpus_bleu(sub_comp, [sub_refs], smooth_method="exp").score
        delta_bleu_samples.append(b_t - b_c)

        c_t = sacrebleu.corpus_chrf(sub_tssa, [sub_refs], word_order=2).score
        c_c = sacrebleu.corpus_chrf(sub_comp, [sub_refs], word_order=2).score
        delta_chrf_samples.append(c_t - c_c)

    delta_bleu_samples = np.array(delta_bleu_samples)
    delta_chrf_samples = np.array(delta_chrf_samples)

    p_bleu = np.mean(delta_bleu_samples <= 0)
    p_chrf = np.mean(delta_chrf_samples <= 0)

    ci_bleu = (np.percentile(delta_bleu_samples, 2.5), np.percentile(delta_bleu_samples, 97.5))
    ci_chrf = (np.percentile(delta_chrf_samples, 2.5), np.percentile(delta_chrf_samples, 97.5))

    return {
        "bleu": {
            "tssa": real_bleu_tssa,
            "comp": real_bleu_comp,
            "delta": real_delta_bleu,
            "p_value": p_bleu,
            "ci_95": ci_bleu
        },
        "chrf": {
            "tssa": real_chrf_tssa,
            "comp": real_chrf_comp,
            "delta": real_delta_chrf,
            "p_value": p_chrf,
            "ci_95": ci_chrf
        }
    }

def format_significance(p):
    if p < 0.001:
        return f"{p:.4f} (p < 0.001) ***", "^(***)"
    elif p < 0.01:
        return f"{p:.4f} (p < 0.01) **", "^(\\dagger)"
    elif p < 0.05:
        return f"{p:.4f} (p < 0.05) *", "^(\\ddagger)"
    else:
        return f"{p:.4f} (n.s.)", ""

def main():
    parser = argparse.ArgumentParser(description="Paired Bootstrap Resampling Significance Test for ViT5")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of bootstrap iterations (default 1000)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    langs = ["rhade", "tay", "bahnaric"]
    competitors = ["align_to_distill", "shift_aet", "awesome_align", "cl_lsa"]

    print("=" * 115)
    print(f"   📊 PAIRED BOOTSTRAP RESAMPLING FOR ViT5 BENCHMARK (B = {args.num_samples}, Seed = {args.seed})")
    print("=" * 115)

    results = []

    for lang in langs:
        tssa_path = os.path.join(args.checkpoints_dir, f"vit5_tssa_{lang}", "test_predictions.csv")
        vanilla_path = os.path.join(args.checkpoints_dir, f"vit5_vanilla_{lang}", "test_predictions.csv")

        _, refs, tssa_preds = load_predictions(tssa_path)
        if not refs:
            print(f"[!] Bỏ qua {lang.upper()}: Không tìm thấy file {tssa_path}")
            continue

        print(f"\n[*] Đang kiểm định cho ngôn ngữ: {lang.upper()} ({len(refs)} câu test)")

        # 1. So với Vanilla ViT5
        _, _, v_preds = load_predictions(vanilla_path)
        if v_preds:
            res_v = paired_bootstrap_test(tssa_preds, v_preds, refs, args.num_samples, args.seed, desc=f"vs. Vanilla ViT5")
            results.append((lang.upper(), "Vanilla ViT5", res_v))

        # 2. Tìm đối thủ tốt nhất trong các baselines ViT5
        best_comp = None
        best_comp_bleu = -1.0
        best_comp_preds = None

        for comp in competitors:
            comp_path = os.path.join(args.checkpoints_dir, f"vit5_{comp}_{lang}", "test_predictions.csv")
            if os.path.exists(comp_path):
                _, _, c_preds = load_predictions(comp_path)
                if c_preds:
                    b_score = sacrebleu.corpus_bleu(c_preds, [refs], smooth_method="exp").score
                    if b_score > best_comp_bleu:
                        best_comp_bleu = b_score
                        best_comp = comp
                        best_comp_preds = c_preds

        if best_comp and best_comp_preds:
            res_s = paired_bootstrap_test(tssa_preds, best_comp_preds, refs, args.num_samples, args.seed, desc=f"vs. Best Comp ({best_comp})")
            results.append((lang.upper(), f"Best Comp ({best_comp})", res_s))

    if not results:
        print("[!] Chưa có đủ dự đoán để tính kiểm định thống kê. Hãy chạy scripts/run_vit5_full_benchmark.sh trước.")
        return

    print(f"\n{'NGÔN NGỮ':<10} | {'ĐỐI THỦ SO SÁNH':<26} | {'METRIC':<7} | {'COMP':<6} | {'TSSA':<6} | {'Δ SCORE':<8} | {'95% CI':<16} | {'P-VALUE':<18}")
    print("-" * 115)

    for lang, comp_name, res in results:
        b = res["bleu"]
        p_str_b, _ = format_significance(b["p_value"])
        ci_str_b = f"[{b['ci_95'][0]:+.2f}, {b['ci_95'][1]:+.2f}]"
        print(f"{lang:<10} | {comp_name:<26} | {'BLEU':<7} | {b['comp']:<6.2f} | {b['tssa']:<6.2f} | {b['delta']:<+8.2f} | {ci_str_b:<16} | {p_str_b:<18}")

        c = res["chrf"]
        p_str_c, _ = format_significance(c["p_value"])
        ci_str_c = f"[{c['ci_95'][0]:+.2f}, {c['ci_95'][1]:+.2f}]"
        print(f"{'':<10} | {'':<26} | {'chrF++':<7} | {c['comp']:<6.2f} | {c['tssa']:<6.2f} | {c['delta']:<+8.2f} | {ci_str_c:<16} | {p_str_c:<18}")
        print("-" * 115)

    print("\n[+] Ghi chú ký hiệu học thuật:")
    print("    ^(\\dagger)  : Ý nghĩa thống kê vượt trội so với Vanilla ViT5 (p < 0.01)")
    print("    ^(\\ddagger) : Ý nghĩa thống kê vượt trội so với Best Competitor on ViT5 (p < 0.05)")

if __name__ == "__main__":
    main()
