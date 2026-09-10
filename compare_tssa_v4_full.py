"""
UniTSSA 4.0 Benchmark Gate: 5-Way Comprehensive Comparison Reporter
Compares newly trained UniTSSA 4.0 models (in checkpoints/tssa_v4/) against:
1. Baseline Vanilla Models (BARTpho & ViT5)
2. Legacy TSSA v1 Models (Archived in checkpoints/backup_tssa_legacy_v1/)
3. TSSA 2.1 Pilot Models (in checkpoints/)
4. TSSA 3.0 Models (in checkpoints/tssa_v3/)
5. UniTSSA 4.0 Models (in checkpoints/tssa_v4/)

Prints ANSI colored tables with delta comparisons and pass verification criteria.
"""

import sys
import os
import json
import sacrebleu
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

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
    print("=" * 140)
    print("     📊 BÁO CÁO ĐỐI SOÁT 5 CHIỀU TOÀN DIỆN: UniTSSA 4.0 vs TSSA 3.0 vs TSSA 2.1 vs TSSA v1 vs VANILLA")
    print("=" * 140)

    languages = ["rhade", "tay", "bahnaric"]
    lang_labels = {"rhade": "Rhade (Ê Đê)", "tay": "Tay (Tày)", "bahnaric": "Bahnar (Ba Na)"}

    rows = []

    # 1. BARTpho Models
    for lang in languages:
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

        v2_metrics = extract_metrics(os.path.join("checkpoints", f"tssa_{lang}"))
        v3_metrics = extract_metrics(os.path.join("checkpoints", "tssa_v3", f"tssa_{lang}"))
        v4_metrics = extract_metrics(os.path.join("checkpoints", "tssa_v4", f"tssa_{lang}"))

        v_bleu = vanilla_data["bleu"]
        l_bleu = legacy_data["bleu"]
        v2_bleu = v2_metrics["bleu"] if v2_metrics else None
        v3_bleu = v3_metrics["bleu"] if v3_metrics else None
        v4_bleu = v4_metrics["bleu"] if v4_metrics else None

        v_chrf = vanilla_data["chrf"]
        l_chrf = legacy_data["chrf"]
        v2_chrf = v2_metrics["chrf"] if v2_metrics else None
        v3_chrf = v3_metrics["chrf"] if v3_metrics else None
        v4_chrf = v4_metrics["chrf"] if v4_metrics else None

        d_vanilla_bleu = (v4_bleu - v_bleu) if v4_bleu is not None else None
        d_v3_bleu = (v4_bleu - v3_bleu) if (v4_bleu is not None and v3_bleu is not None) else None

        d_vanilla_chrf = (v4_chrf - v_chrf) if v4_chrf is not None else None
        d_v3_chrf = (v4_chrf - v3_chrf) if (v4_chrf is not None and v3_chrf is not None) else None

        rows.append({
            "backbone": "BARTpho",
            "lang": lang_labels[lang],
            "vanilla_bleu": v_bleu,
            "legacy_bleu": l_bleu,
            "v2_bleu": v2_bleu,
            "v3_bleu": v3_bleu,
            "v4_bleu": v4_bleu,
            "d_vanilla_bleu": d_vanilla_bleu,
            "d_v3_bleu": d_v3_bleu,
            "vanilla_chrf": v_chrf,
            "legacy_chrf": l_chrf,
            "v2_chrf": v2_chrf,
            "v3_chrf": v3_chrf,
            "v4_chrf": v4_chrf,
            "d_vanilla_chrf": d_vanilla_chrf,
            "d_v3_chrf": d_v3_chrf,
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

        v2_metrics = extract_metrics(os.path.join("checkpoints", f"vit5_tssa_{lang}"))
        v3_metrics = extract_metrics(os.path.join("checkpoints", "tssa_v3", f"vit5_tssa_{lang}"))
        v4_metrics = extract_metrics(os.path.join("checkpoints", "tssa_v4", f"vit5_tssa_{lang}"))

        v_bleu = vanilla_data["bleu"]
        l_bleu = legacy_data["bleu"]
        v2_bleu = v2_metrics["bleu"] if v2_metrics else None
        v3_bleu = v3_metrics["bleu"] if v3_metrics else None
        v4_bleu = v4_metrics["bleu"] if v4_metrics else None

        v_chrf = vanilla_data["chrf"]
        l_chrf = legacy_data["chrf"]
        v2_chrf = v2_metrics["chrf"] if v2_metrics else None
        v3_chrf = v3_metrics["chrf"] if v3_metrics else None
        v4_chrf = v4_metrics["chrf"] if v4_metrics else None

        d_vanilla_bleu = (v4_bleu - v_bleu) if v4_bleu is not None else None
        d_v3_bleu = (v4_bleu - v3_bleu) if (v4_bleu is not None and v3_bleu is not None) else None

        d_vanilla_chrf = (v4_chrf - v_chrf) if v4_chrf is not None else None
        d_v3_chrf = (v4_chrf - v3_chrf) if (v4_chrf is not None and v3_chrf is not None) else None

        rows.append({
            "backbone": "ViT5",
            "lang": lang_labels[lang],
            "vanilla_bleu": v_bleu,
            "legacy_bleu": l_bleu,
            "v2_bleu": v2_bleu,
            "v3_bleu": v3_bleu,
            "v4_bleu": v4_bleu,
            "d_vanilla_bleu": d_vanilla_bleu,
            "d_v3_bleu": d_v3_bleu,
            "vanilla_chrf": v_chrf,
            "legacy_chrf": l_chrf,
            "v2_chrf": v2_chrf,
            "v3_chrf": v3_chrf,
            "v4_chrf": v4_chrf,
            "d_vanilla_chrf": d_vanilla_chrf,
            "d_v3_chrf": d_v3_chrf,
        })

    # Print SacreBLEU Table
    print("\n--- [BẢNG 1: SacreBLEU (ĐỘ ĐO CHÍNH - NAACL SPEC)] ---")
    header_bleu = f"{'MÔ HÌNH':<9} | {'NGÔN NGỮ':<15} | {'VANILLA':<8} | {'TSSA v1':<8} | {'TSSA 2.1':<8} | {'TSSA 3.0':<8} | {'UniTSSA 4.0':<11} | {'Δ vs VANILLA':<13} | {'TRẠNG THÁI'}"
    print(header_bleu)
    print("-" * 140)

    all_win = True
    any_done = False

    for r in rows:
        v2_str = f"{r['v2_bleu']:.2f}" if r['v2_bleu'] is not None else "--"
        v3_str = f"{r['v3_bleu']:.2f}" if r['v3_bleu'] is not None else "--"
        v4_str = f"{r['v4_bleu']:.2f}" if r['v4_bleu'] is not None else "Đang chờ"
        dv_str = f"{r['d_vanilla_bleu']:+.2f}" if r['d_vanilla_bleu'] is not None else "--"

        status = "⏳ Chưa có số liệu"
        if r['v4_bleu'] is not None:
            any_done = True
            if r['d_vanilla_bleu'] >= 0.20:
                status = "✅ THẮNG ĐẬM (>= +0.2)"
            elif r['d_vanilla_bleu'] > 0:
                status = "✅ THẮNG (Tăng vs Vanilla)"
            else:
                status = "❌ THUA vs Vanilla"
                all_win = False
        else:
            all_win = False

        print(f"{r['backbone']:<9} | {r['lang']:<15} | {r['vanilla_bleu']:<8.2f} | {r['legacy_bleu']:<8.2f} | {v2_str:<8} | {v3_str:<8} | {v4_str:<11} | {dv_str:<13} | {status}")

    print("-" * 140)

    if any_done and all_win:
        print("\n🎉 [KẾT QUẢ TỔNG THỂ] 100% CẢ 6 MÔ HÌNH ĐỀU THẮNG VANILLA BASELINE (ALL GREEN)!")
    elif any_done:
        print("\n[*] Đang trong quá trình hoàn thiện các mô hình còn lại.")

    # Print chrF++ Table
    print("\n--- [BẢNG 2: chrF++] ---")
    header_chrf = f"{'MÔ HÌNH':<9} | {'NGÔN NGỮ':<15} | {'VANILLA':<8} | {'TSSA v1':<8} | {'TSSA 2.1':<8} | {'TSSA 3.0':<8} | {'UniTSSA 4.0':<11} | {'Δ vs VANILLA':<13} | {'TRẠNG THÁI'}"
    print(header_chrf)
    print("-" * 140)

    for r in rows:
        v2_str = f"{r['v2_chrf']:.2f}" if r['v2_chrf'] is not None else "--"
        v3_str = f"{r['v3_chrf']:.2f}" if r['v3_chrf'] is not None else "--"
        v4_str = f"{r['v4_chrf']:.2f}" if r['v4_chrf'] is not None else "Đang chờ"
        dv_str = f"{r['d_vanilla_chrf']:+.2f}" if r['d_vanilla_chrf'] is not None else "--"

        status = "⏳ Chưa có số liệu"
        if r['v4_chrf'] is not None:
            if r['d_vanilla_chrf'] > 0:
                status = "✅ Tăng chrF++"
            else:
                status = "❌ Giảm chrF++"

        print(f"{r['backbone']:<9} | {r['lang']:<15} | {r['vanilla_chrf']:<8.2f} | {r['legacy_chrf']:<8.2f} | {v2_str:<8} | {v3_str:<8} | {v4_str:<11} | {dv_str:<13} | {status}")

    print("-" * 140)

if __name__ == "__main__":
    main()
