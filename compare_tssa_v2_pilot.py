"""
TSSA 2.1 Verification Gate: Comprehensive Comparison Reporter
Compares newly trained TSSA 2.1 models against:
1. Baseline Vanilla Models (BARTpho & ViT5)
2. Legacy TSSA v1 Models (Archived in backup_tssa_legacy_v1/ or Official Table)
Prints formatted tables with delta scores and gate status.
"""

import os
import json
import sacrebleu
import pandas as pd

OFFICIAL_LEGACY_REFS = {
    "bartpho": {
        "vanilla": {
            "rhade": {"bleu": 34.02, "chrf": 50.41},
            "tay": {"bleu": 40.54, "chrf": 57.06},
            "bahnaric": {"bleu": 26.65, "chrf": 44.59}
        },
        "legacy_tssa": {
            "rhade": {"bleu": 34.86, "chrf": 51.52},
            "tay": {"bleu": 42.12, "chrf": 58.74},
            "bahnaric": {"bleu": 26.68, "chrf": 44.82}
        }
    },
    "vit5": {
        "vanilla": {
            "rhade": {"bleu": 30.28, "chrf": 46.47},
            "tay": {"bleu": 35.84, "chrf": 53.64},
            "bahnaric": {"bleu": 24.32, "chrf": 42.34}
        },
        "legacy_tssa": {
            "rhade": {"bleu": 30.12, "chrf": 46.25},
            "tay": {"bleu": 36.31, "chrf": 54.02},
            "bahnaric": {"bleu": 24.16, "chrf": 42.15}
        }
    }
}

def extract_metrics(ckpt_path):
    if not os.path.exists(ckpt_path):
        return None
    metrics_file = os.path.join(ckpt_path, "eval_metrics.json")
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    pred_file = os.path.join(ckpt_path, "test_predictions.csv")
    if os.path.exists(pred_file):
        try:
            df = pd.read_csv(pred_file)
            ref_col, pred_col = None, None
            for c in df.columns:
                cl = c.lower()
                if "ref" in cl or "target" in cl:
                    ref_col = c
                elif "pred" in cl or "translation" in cl or "hyp" in cl:
                    pred_col = c
            if ref_col and pred_col:
                refs = [str(r).strip() for r in df[ref_col].fillna("").tolist()]
                preds = [str(p).strip() for p in df[pred_col].fillna("").tolist()]
                b = sacrebleu.corpus_bleu(preds, [refs], smooth_method="exp").score
                c = sacrebleu.corpus_chrf(preds, [refs], word_order=2).score
                return {"bleu": round(b, 2), "chrf": round(c, 2)}
        except Exception:
            pass
    return None

def main():
    print("=" * 105)
    print("         📊 BÁO CÁO ĐỐI SOÁT TỔNG THỂ: TSSA 2.1 vs TSSA v1 (CŨ) vs VANILLA BASELINE")
    print("=" * 105)

    languages = ["rhade", "tay", "bahnaric"]
    lang_labels = {"rhade": "Rhade (Ê Đê)", "tay": "Tay (Tày)", "bahnaric": "Bahnar (Ba Na)"}

    rows = []

    # 1. BARTpho Models
    for lang in languages:
        vanilla_data = OFFICIAL_LEGACY_REFS["bartpho"]["vanilla"][lang]
        legacy_data = OFFICIAL_LEGACY_REFS["bartpho"]["legacy_tssa"][lang]

        # Check backup legacy dir if exists
        backup_metrics = extract_metrics(os.path.join("checkpoints", "backup_tssa_legacy_v1", f"tssa_{lang}"))
        if backup_metrics and "bleu" in backup_metrics:
            legacy_data = backup_metrics

        # New TSSA 2.1
        new_metrics = extract_metrics(os.path.join("checkpoints", f"tssa_{lang}"))

        v_bleu = vanilla_data["bleu"]
        l_bleu = legacy_data["bleu"]
        n_bleu = new_metrics["bleu"] if new_metrics else None

        v_chrf = vanilla_data["chrf"]
        l_chrf = legacy_data["chrf"]
        n_chrf = new_metrics["chrf"] if new_metrics else None

        d_vanilla_bleu = (n_bleu - v_bleu) if n_bleu is not None else None
        d_legacy_bleu = (n_bleu - l_bleu) if n_bleu is not None else None

        rows.append({
            "backbone": "BARTpho",
            "lang": lang_labels[lang],
            "vanilla_bleu": v_bleu,
            "legacy_bleu": l_bleu,
            "new_bleu": n_bleu,
            "d_vanilla": d_vanilla_bleu,
            "d_legacy": d_legacy_bleu,
            "vanilla_chrf": v_chrf,
            "legacy_chrf": l_chrf,
            "new_chrf": n_chrf
        })

    # 2. ViT5 Models
    for lang in languages:
        vanilla_data = OFFICIAL_LEGACY_REFS["vit5"]["vanilla"][lang]
        legacy_data = OFFICIAL_LEGACY_REFS["vit5"]["legacy_tssa"][lang]

        backup_metrics = extract_metrics(os.path.join("checkpoints", "backup_tssa_legacy_v1", f"vit5_tssa_{lang}"))
        if backup_metrics and "bleu" in backup_metrics:
            legacy_data = backup_metrics

        new_metrics = extract_metrics(os.path.join("checkpoints", f"vit5_tssa_{lang}"))

        v_bleu = vanilla_data["bleu"]
        l_bleu = legacy_data["bleu"]
        n_bleu = new_metrics["bleu"] if new_metrics else None

        v_chrf = vanilla_data["chrf"]
        l_chrf = legacy_data["chrf"]
        n_chrf = new_metrics["chrf"] if new_metrics else None

        d_vanilla_bleu = (n_bleu - v_bleu) if n_bleu is not None else None
        d_legacy_bleu = (n_bleu - l_bleu) if n_bleu is not None else None

        rows.append({
            "backbone": "ViT5",
            "lang": lang_labels[lang],
            "vanilla_bleu": v_bleu,
            "legacy_bleu": l_bleu,
            "new_bleu": n_bleu,
            "d_vanilla": d_vanilla_bleu,
            "d_legacy": d_legacy_bleu,
            "vanilla_chrf": v_chrf,
            "legacy_chrf": l_chrf,
            "new_chrf": n_chrf
        })

    # Print Table
    header = f"{'MÔ HÌNH':<9} | {'NGÔN NGỮ':<15} | {'VANILLA':<8} | {'TSSA v1':<8} | {'TSSA 2.1':<8} | {'Δ vs VANILLA':<13} | {'Δ vs TSSA v1':<13} | {'TRẠNG THÁI'}"
    print(header)
    print("-" * 105)

    all_passed = True
    any_evaluated = False

    for r in rows:
        n_str = f"{r['new_bleu']:.2f}" if r['new_bleu'] is not None else "Đang chờ"
        dv_str = f"{r['d_vanilla']:+.2f}" if r['d_vanilla'] is not None else "--"
        dl_str = f"{r['d_legacy']:+.2f}" if r['d_legacy'] is not None else "--"

        status = "⏳ Chưa có số liệu"
        if r['new_bleu'] is not None:
            any_evaluated = True
            # Success criterion: higher than vanilla AND higher than legacy
            if r['d_vanilla'] > 0 and r['d_legacy'] >= 0:
                status = "✅ VƯỢT TRỘI (Đạt)"
            elif r['d_vanilla'] > 0:
                status = "⚠️ Tăng vs Vanilla, sát TSSA v1"
            else:
                status = "❌ Thụt lùi vs Vanilla"
                all_passed = False

        print(f"{r['backbone']:<9} | {r['lang']:<15} | {r['vanilla_bleu']:<8.2f} | {r['legacy_bleu']:<8.2f} | {n_str:<8} | {dv_str:<13} | {dl_str:<13} | {status}")

    print("-" * 105)
    print("\n[!] Tiêu chí kiểm định Verification Gate:")
    print("    1. TSSA 2.1 phải vượt trội hơn Vanilla trên CẢ 6 MÔ HÌNH (đặc biệt ViT5 Rhade & Bahnaric).")
    print("    2. Ba Na trên BARTpho phải tạo ra khoảng cách đáng kể (> +0.50 BLEU so với Vanilla 26.65).")
    print("    3. p-value trong paired bootstrap significance phải đạt p < 0.05 đối với Vanilla.")
    print("=" * 105)

if __name__ == "__main__":
    main()
