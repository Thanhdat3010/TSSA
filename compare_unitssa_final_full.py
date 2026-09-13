"""
UniTSSA Final Benchmark Gate: 8-Way Comprehensive Comparison Reporter (NAACL Gold Standard)
Compares newly trained UniTSSA Final models (in checkpoints/tssa_final/) against:
1. Baseline Vanilla Models (BARTpho & ViT5)
2. Legacy TSSA v1 Models (Archived in checkpoints/backup_tssa_legacy_v1/)
3. TSSA 2.1 Pilot Models (in checkpoints/)
4. TSSA 3.0 Models (in checkpoints/tssa_v3/)
5. UniTSSA 4.0 Models (in checkpoints/tssa_v4/)
6. UniTSSA 5.0 Models (in checkpoints/tssa_v5/)
7. UniTSSA 6.0 Models (in checkpoints/tssa_v6/)
8. UniTSSA Final Models (in checkpoints/tssa_final/)

Prints formatted comparison tables with delta comparisons and pass verification criteria.
"""

import sys
import os
import json

try:
    import sacrebleu
except ImportError:
    sacrebleu = None

try:
    import pandas as pd
except ImportError:
    pd = None

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Official reference numbers across all completed generations (5 epochs, seed 42)
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
        },
        "tssa_v2": {
            "rhade": {"bleu": 24.11, "chrf": 40.39},
            "tay": {"bleu": 25.40, "chrf": 36.28},
            "bahnaric": {"bleu": 9.37, "chrf": 23.74}
        },
        "tssa_v3": {
            "rhade": {"bleu": 24.34, "chrf": 40.60},
            "tay": {"bleu": 25.68, "chrf": 36.50},
            "bahnaric": {"bleu": 9.26, "chrf": 23.82}
        },
        "unitssa_v4": {
            "rhade": {"bleu": 24.26, "chrf": 40.40},
            "tay": {"bleu": 25.35, "chrf": 36.28},
            "bahnaric": {"bleu": 9.30, "chrf": 23.71}
        },
        "unitssa_v5": {
            "rhade": {"bleu": 24.09, "chrf": 40.38},
            "tay": {"bleu": 25.56, "chrf": 36.47},
            "bahnaric": {"bleu": 9.39, "chrf": 23.63}
        },
        "unitssa_v6": {
            "rhade": {"bleu": 24.16, "chrf": 40.50},
            "tay": {"bleu": 25.34, "chrf": 36.23},
            "bahnaric": {"bleu": 9.18, "chrf": 23.52}
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
        },
        "tssa_v2": {
            "rhade": {"bleu": 30.20, "chrf": 46.43},
            "tay": {"bleu": 34.97, "chrf": 45.06},
            "bahnaric": {"bleu": 11.36, "chrf": 27.56}
        },
        "tssa_v3": {
            "rhade": {"bleu": 30.14, "chrf": 46.37},
            "tay": {"bleu": 34.78, "chrf": 44.89},
            "bahnaric": {"bleu": 11.00, "chrf": 27.18}
        },
        "unitssa_v4": {
            "rhade": {"bleu": 30.49, "chrf": 46.57},
            "tay": {"bleu": 35.14, "chrf": 45.39},
            "bahnaric": {"bleu": 11.17, "chrf": 27.53}
        },
        "unitssa_v5": {
            "rhade": {"bleu": 30.12, "chrf": 46.55},
            "tay": {"bleu": 34.81, "chrf": 44.75},
            "bahnaric": {"bleu": 11.26, "chrf": 27.52}
        },
        "unitssa_v6": {
            "rhade": {"bleu": 30.10, "chrf": 46.45},
            "tay": {"bleu": 34.67, "chrf": 44.94},
            "bahnaric": {"bleu": 10.75, "chrf": 27.08}
        }
    }
}

def extract_metrics(ckpt_dir: str):
    """Extracts evaluation metrics (BLEU, chrF++) from saved test_predictions.csv."""
    pred_path = os.path.join(ckpt_dir, "test_predictions.csv")
    if not os.path.exists(pred_path) or pd is None or sacrebleu is None:
        return None

    try:
        df = pd.read_csv(pred_path)
        if "prediction" not in df.columns or "target" not in df.columns:
            return None

        preds = [str(p) if pd.notna(p) else "" for p in df["prediction"].tolist()]
        refs = [[str(r) if pd.notna(r) else "" for r in df["target"].tolist()]]

        bleu = sacrebleu.corpus_bleu(preds, refs, smooth_method="exp").score
        chrf = sacrebleu.corpus_chrf(preds, refs).score

        return {
            "bleu": round(float(bleu), 2),
            "chrf": round(float(chrf), 2)
        }
    except Exception:
        return None

def main():
    print("=" * 160)
    print("     📊 BÁO CÁO ĐỐI SOÁT 8 CHIỀU TOÀN DIỆN: UniTSSA Final vs v6.0 vs v5.0 vs v4.0 vs v3.0 vs v2.1 vs v1 vs VANILLA")
    print("=" * 160)

    languages = [
        ("rhade", "Rhade (Ê Đê)"),
        ("tay", "Tay (Tày)"),
        ("bahnaric", "Bahnar (Ba Na)")
    ]

    models = [
        ("BARTpho", "bartpho", "tssa_"),
        ("ViT5", "vit5", "vit5_tssa_")
    ]

    all_data = []

    for arch_name, arch_key, prefix in models:
        for lang_code, lang_name in languages:
            row = {
                "arch": arch_name,
                "arch_key": arch_key,
                "lang_code": lang_code,
                "lang_name": lang_name,
                "vanilla_bleu": OFFICIAL_LEGACY_REFS[arch_key]["vanilla"][lang_code]["bleu"],
                "vanilla_chrf": OFFICIAL_LEGACY_REFS[arch_key]["vanilla"][lang_code]["chrf"],
                "v1_bleu": OFFICIAL_LEGACY_REFS[arch_key]["legacy_tssa"][lang_code]["bleu"],
                "v1_chrf": OFFICIAL_LEGACY_REFS[arch_key]["legacy_tssa"][lang_code]["chrf"],
                "v2_bleu": OFFICIAL_LEGACY_REFS[arch_key]["tssa_v2"][lang_code]["bleu"],
                "v2_chrf": OFFICIAL_LEGACY_REFS[arch_key]["tssa_v2"][lang_code]["chrf"],
                "v3_bleu": OFFICIAL_LEGACY_REFS[arch_key]["tssa_v3"][lang_code]["bleu"],
                "v3_chrf": OFFICIAL_LEGACY_REFS[arch_key]["tssa_v3"][lang_code]["chrf"],
                "v4_bleu": OFFICIAL_LEGACY_REFS[arch_key]["unitssa_v4"][lang_code]["bleu"],
                "v4_chrf": OFFICIAL_LEGACY_REFS[arch_key]["unitssa_v4"][lang_code]["chrf"],
                "v5_bleu": OFFICIAL_LEGACY_REFS[arch_key]["unitssa_v5"][lang_code]["bleu"],
                "v5_chrf": OFFICIAL_LEGACY_REFS[arch_key]["unitssa_v5"][lang_code]["chrf"],
                "v6_bleu": OFFICIAL_LEGACY_REFS[arch_key]["unitssa_v6"][lang_code]["bleu"],
                "v6_chrf": OFFICIAL_LEGACY_REFS[arch_key]["unitssa_v6"][lang_code]["chrf"],
            }

            # Check v6 dynamically if available
            v6_ckpt = os.path.join("checkpoints", "tssa_v6", f"{prefix}{lang_code}")
            v6_m = extract_metrics(v6_ckpt)
            if v6_m:
                row["v6_bleu"] = v6_m["bleu"]
                row["v6_chrf"] = v6_m["chrf"]

            # Check final
            final_ckpt = os.path.join("checkpoints", "tssa_final", f"{prefix}{lang_code}")
            final_m = extract_metrics(final_ckpt)
            if final_m:
                row["final_bleu"] = final_m["bleu"]
                row["final_chrf"] = final_m["chrf"]
            else:
                row["final_bleu"] = None
                row["final_chrf"] = None

            all_data.append(row)

    # -------------------------------------------------------------------------
    # BẢNG 1: SacreBLEU
    # -------------------------------------------------------------------------
    print("\n--- [BẢNG 1: SacreBLEU (ĐỘ ĐO CHÍNH - NAACL SPEC)] ---")
    header_b1 = (
        f"{'MÔ HÌNH':<8} | {'NGÔN NGỮ':<15} | {'VANILLA':<8} | {'v1':<6} | {'v2.1':<6} | {'v3.0':<6} | "
        f"{'v4.0':<6} | {'v5.0':<6} | {'v6.0':<6} | {'FINAL':<8} | {'Δ vs VANILLA':<13} | {'TRẠNG THÁI'}"
    )
    print(header_b1)
    print("-" * 140)

    completed_count = 0
    win_count = 0

    for r in all_data:
        vanilla = r["vanilla_bleu"]
        v1 = r["v1_bleu"]
        v2 = r["v2_bleu"]
        v3 = r["v3_bleu"]
        v4 = r["v4_bleu"]
        v5 = r["v5_bleu"]
        v6 = r["v6_bleu"]
        f_bleu = r["final_bleu"]

        if f_bleu is not None:
            completed_count += 1
            delta = f_bleu - vanilla
            d_str = f"{delta:+6.2f}"
            if delta >= 0.20:
                status = "✅ THẮNG ĐẬM (>= +0.2)"
                win_count += 1
            elif delta >= 0.00:
                status = "✅ THẮNG / NGANG HÀNG"
                win_count += 1
            else:
                status = "❌ THUA vs Vanilla"
            f_str = f"{f_bleu:<8.2f}"
        else:
            f_str = f"{'Đang chờ':<8}"
            d_str = f"{'--':<13}"
            status = "⏳ Đang chạy..."

        print(
            f"{r['arch']:<8} | {r['lang_name']:<15} | {vanilla:<8.2f} | {v1:<6.2f} | {v2:<6.2f} | {v3:<6.2f} | "
            f"{v4:<6.2f} | {v5:<6.2f} | {v6:<6.2f} | {f_str} | {d_str:<13} | {status}"
        )

    print("-" * 140)

    # -------------------------------------------------------------------------
    # BẢNG 2: chrF++
    # -------------------------------------------------------------------------
    print("\n--- [BẢNG 2: chrF++] ---")
    header_b2 = (
        f"{'MÔ HÌNH':<8} | {'NGÔN NGỮ':<15} | {'VANILLA':<8} | {'v1':<6} | {'v2.1':<6} | {'v3.0':<6} | "
        f"{'v4.0':<6} | {'v5.0':<6} | {'v6.0':<6} | {'FINAL':<8} | {'Δ vs VANILLA':<13} | {'TRẠNG THÁI'}"
    )
    print(header_b2)
    print("-" * 140)

    for r in all_data:
        vanilla = r["vanilla_chrf"]
        v1 = r["v1_chrf"]
        v2 = r["v2_chrf"]
        v3 = r["v3_chrf"]
        v4 = r["v4_chrf"]
        v5 = r["v5_chrf"]
        v6 = r["v6_chrf"]
        f_chrf = r["final_chrf"]

        if f_chrf is not None:
            delta = f_chrf - vanilla
            d_str = f"{delta:+6.2f}"
            status = "✅ Tăng chrF++" if delta >= 0 else "❌ Giảm chrF++"
            f_str = f"{f_chrf:<8.2f}"
        else:
            f_str = f"{'Đang chờ':<8}"
            d_str = f"{'--':<13}"
            status = "⏳ Đang chạy..."

        print(
            f"{r['arch']:<8} | {r['lang_name']:<15} | {vanilla:<8.2f} | {v1:<6.2f} | {v2:<6.2f} | {v3:<6.2f} | "
            f"{v4:<6.2f} | {v5:<6.2f} | {v6:<6.2f} | {f_str} | {d_str:<13} | {status}"
        )

    print("-" * 140)

    if completed_count == 6:
        print("\n========================================================================")
        print(f"    🎉 HOÀN TẤT TOÀN BỘ 6 MÔ HÌNH UniTSSA FINAL! TỶ LỆ THẮNG: {win_count}/6")
        print("========================================================================")
    else:
        print(f"\n[*] Tiến trình UniTSSA Final: Đã hoàn tất {completed_count}/6 mô hình.")

if __name__ == "__main__":
    main()
