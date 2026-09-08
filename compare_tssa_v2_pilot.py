"""
TSSA 2.1 Verification Gate: Comprehensive Comparison Reporter
Compares newly trained TSSA 2.1 models against:
1. Baseline Vanilla Models (BARTpho & ViT5)
2. Legacy TSSA v1 Models (Archived in backup_tssa_legacy_v1/ or Official Results Table)
Prints formatted tables with delta scores and gate status.
"""

import os
import json
import sacrebleu
import pandas as pd

# Official numbers from docs/OFFICIAL_EXPERIMENT_RESULTS.md (5 epochs, seed 42)
OFFICIAL_LEGACY_REFS = {
    "bartpho": {
        "vanilla": {
            "rhade": {"bleu": 23.41, "chrf": 39.33},
            "tay": {"bleu": 24.67, "chrf": 35.74},
            "bahnaric": {"bleu": 9.63, "chrf": 23.47}
        },
        "legacy_tssa": {
            "rhade": {"bleu": 24.11, "chrf": 40.43},
            "tay": {"bleu": 25.46, "chrf": 36.31},
            "bahnaric": {"bleu": 9.66, "chrf": 23.89}
        }
    },
    "vit5": {
        "vanilla": {
            "rhade": {"bleu": 30.28, "chrf": 46.47},
            "tay": {"bleu": 34.99, "chrf": 44.72},
            "bahnaric": {"bleu": 11.34, "chrf": 27.67}
        },
        "legacy_tssa": {
            "rhade": {"bleu": 30.08, "chrf": 46.48},
            "tay": {"bleu": 35.44, "chrf": 45.21},
            "bahnaric": {"bleu": 11.00, "chrf": 27.71}
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
    print("=" * 115)
    print("         📊 BÁO CÁO ĐỐI SOÁT TỔNG THỂ: TSSA 2.1 vs TSSA v1 (CŨ) vs VANILLA BASELINE")
    print("=" * 115)

    languages = ["rhade", "tay", "bahnaric"]
    lang_labels = {"rhade": "Rhade (Ê Đê)", "tay": "Tay (Tày)", "bahnaric": "Bahnar (Ba Na)"}

    rows = []

    # 1. BARTpho Models
    for lang in languages:
        # Dynamic check for Vanilla BARTpho on disk
        vanilla_data = OFFICIAL_LEGACY_REFS["bartpho"]["vanilla"][lang]
        for v_dir in [f"checkpoints/bartpho_vanilla_{lang}", f"checkpoints/vanilla_{lang}"]:
            v_met = extract_metrics(v_dir)
            if v_met and "bleu" in v_met:
                vanilla_data = v_met
                break

        legacy_data = OFFICIAL_LEGACY_REFS["bartpho"]["legacy_tssa"][lang]
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

        d_vanilla_chrf = (n_chrf - v_chrf) if n_chrf is not None else None
        d_legacy_chrf = (n_chrf - l_chrf) if n_chrf is not None else None

        rows.append({
            "backbone": "BARTpho",
            "lang": lang_labels[lang],
            "vanilla_bleu": v_bleu,
            "legacy_bleu": l_bleu,
            "new_bleu": n_bleu,
            "d_vanilla_bleu": d_vanilla_bleu,
            "d_legacy_bleu": d_legacy_bleu,
            "vanilla_chrf": v_chrf,
            "legacy_chrf": l_chrf,
            "new_chrf": n_chrf,
            "d_vanilla_chrf": d_vanilla_chrf,
            "d_legacy_chrf": d_legacy_chrf,
        })

    # 2. ViT5 Models
    for lang in languages:
        vanilla_data = OFFICIAL_LEGACY_REFS["vit5"]["vanilla"][lang]
        for v_dir in [f"checkpoints/vit5_vanilla_{lang}"]:
            v_met = extract_metrics(v_dir)
            if v_met and "bleu" in v_met:
                vanilla_data = v_met
                break

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

        d_vanilla_chrf = (n_chrf - v_chrf) if n_chrf is not None else None
        d_legacy_chrf = (n_chrf - l_chrf) if n_chrf is not None else None

        rows.append({
            "backbone": "ViT5",
            "lang": lang_labels[lang],
            "vanilla_bleu": v_bleu,
            "legacy_bleu": l_bleu,
            "new_bleu": n_bleu,
            "d_vanilla_bleu": d_vanilla_bleu,
            "d_legacy_bleu": d_legacy_bleu,
            "vanilla_chrf": v_chrf,
            "legacy_chrf": l_chrf,
            "new_chrf": n_chrf,
            "d_vanilla_chrf": d_vanilla_chrf,
            "d_legacy_chrf": d_legacy_chrf,
        })

    # Print SacreBLEU Table
    print("\n--- [BẢNG 1: SacreBLEU] ---")
    header_bleu = f"{'MÔ HÌNH':<9} | {'NGÔN NGỮ':<15} | {'VANILLA':<8} | {'TSSA v1':<8} | {'TSSA 2.1':<8} | {'Δ vs VANILLA':<13} | {'Δ vs TSSA v1':<13} | {'ĐÁNH GIÁ'}"
    print(header_bleu)
    print("-" * 115)

    for r in rows:
        n_str = f"{r['new_bleu']:.2f}" if r['new_bleu'] is not None else "Đang chờ"
        dv_str = f"{r['d_vanilla_bleu']:+.2f}" if r['d_vanilla_bleu'] is not None else "--"
        dl_str = f"{r['d_legacy_bleu']:+.2f}" if r['d_legacy_bleu'] is not None else "--"

        status = "⏳ Chưa có số liệu"
        if r['new_bleu'] is not None:
            if r['d_vanilla_bleu'] > 0 and r['d_legacy_bleu'] >= 0:
                status = "✅ VƯỢT TRỘI (Đạt)"
            elif r['d_vanilla_bleu'] > 0:
                status = "✅ Tăng vs Vanilla"
            else:
                status = "❌ Thụt lùi vs Vanilla"

        print(f"{r['backbone']:<9} | {r['lang']:<15} | {r['vanilla_bleu']:<8.2f} | {r['legacy_bleu']:<8.2f} | {n_str:<8} | {dv_str:<13} | {dl_str:<13} | {status}")

    print("-" * 115)

    # Print chrF++ Table
    print("\n--- [BẢNG 2: chrF++] ---")
    header_chrf = f"{'MÔ HÌNH':<9} | {'NGÔN NGỮ':<15} | {'VANILLA':<8} | {'TSSA v1':<8} | {'TSSA 2.1':<8} | {'Δ vs VANILLA':<13} | {'Δ vs TSSA v1':<13} | {'ĐÁNH GIÁ'}"
    print(header_chrf)
    print("-" * 115)

    for r in rows:
        n_str = f"{r['new_chrf']:.2f}" if r['new_chrf'] is not None else "Đang chờ"
        dv_str = f"{r['d_vanilla_chrf']:+.2f}" if r['d_vanilla_chrf'] is not None else "--"
        dl_str = f"{r['d_legacy_chrf']:+.2f}" if r['d_legacy_chrf'] is not None else "--"

        status = "⏳ Chưa có số liệu"
        if r['new_chrf'] is not None:
            if r['d_vanilla_chrf'] > 0 and r['d_legacy_chrf'] >= 0:
                status = "✅ VƯỢT TRỘI (Đạt)"
            elif r['d_vanilla_chrf'] > 0:
                status = "✅ Tăng vs Vanilla"
            else:
                status = "❌ Thụt lùi vs Vanilla"

        print(f"{r['backbone']:<9} | {r['lang']:<15} | {r['vanilla_chrf']:<8.2f} | {r['legacy_chrf']:<8.2f} | {n_str:<8} | {dv_str:<13} | {dl_str:<13} | {status}")

    print("-" * 115)

if __name__ == "__main__":
    main()
