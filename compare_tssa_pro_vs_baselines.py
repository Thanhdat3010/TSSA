"""
compare_tssa_pro_vs_baselines.py
Automated Benchmark & Comparison Reporter for TSSA-Pro (6 Models)
Extracts metrics from checkpoints/tssa_pro/ and performs 3-way comparative analysis against:
1. Vanilla Backbones (BARTpho & ViT5)
2. Strongest Competitor Baselines (AWESOME-align, A2D, CL-LSA, Shift-AET)
3. Previous UniTSSA Final Models
Saves formatted comparison tables to checkpoints/tssa_pro/COMPARISON_REPORT.txt
"""

import os
import sys
import json

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Official Baseline References from OFFICIAL_EXPERIMENT_RESULTS.md
OFFICIAL_REFS = {
    "bartpho": {
        "rhade": {
            "vanilla": {"bleu": 23.41, "chrf": 39.33, "meteor": 33.91, "comet": -0.3678},
            "strongest": {"name": "AWESOME-align", "bleu": 23.05, "chrf": 38.93, "meteor": 33.45, "comet": -0.3841},
            "tssa_final": {"bleu": 24.01, "chrf": 40.27, "meteor": 34.80, "comet": -0.3271}
        },
        "tay": {
            "vanilla": {"bleu": 24.67, "chrf": 35.74, "meteor": 25.68, "comet": -0.5291},
            "strongest": {"name": "AWESOME-align", "bleu": 25.20, "chrf": 36.30, "meteor": 26.44, "comet": -0.5051},
            "tssa_final": {"bleu": 25.32, "chrf": 36.18, "meteor": 26.26, "comet": -0.5046}
        },
        "bahnaric": {
            "vanilla": {"bleu": 9.63, "chrf": 23.47, "meteor": 18.15, "comet": -0.8507},
            "strongest": {"name": "Align-to-Distill", "bleu": 9.15, "chrf": 23.17, "meteor": 18.05, "comet": -0.8598},
            "tssa_final": {"bleu": 9.06, "chrf": 23.37, "meteor": 18.10, "comet": -0.8574}
        }
    },
    "vit5": {
        "rhade": {
            "vanilla": {"bleu": 30.28, "chrf": 46.47, "meteor": 41.09, "comet": -0.0807},
            "strongest": {"name": "AWESOME-align", "bleu": 29.96, "chrf": 46.12, "meteor": 40.94, "comet": -0.1064},
            "tssa_final": {"bleu": 30.64, "chrf": 46.88, "meteor": 41.38, "comet": -0.0841}
        },
        "tay": {
            "vanilla": {"bleu": 34.99, "chrf": 44.72, "meteor": 35.93, "comet": -0.2031},
            "strongest": {"name": "CL-LSA", "bleu": 35.83, "chrf": 45.49, "meteor": 36.51, "comet": -0.1827},
            "tssa_final": {"bleu": 35.97, "chrf": 45.42, "meteor": 36.31, "comet": -0.1798}
        },
        "bahnaric": {
            "vanilla": {"bleu": 11.34, "chrf": 27.67, "meteor": 24.10, "comet": -0.7609},
            "strongest": {"name": "AWESOME-align", "bleu": 11.53, "chrf": 27.92, "meteor": 24.35, "comet": -0.7550},
            "tssa_final": {"bleu": 10.56, "chrf": 27.25, "meteor": 24.41, "comet": -0.7575}
        }
    }
}

def extract_metrics_from_ckpt(ckpt_dir: str):
    """Reads evaluation metrics from eval_metrics.json in checkpoint dir."""
    metrics_path = os.path.join(ckpt_dir, "eval_metrics.json")
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                b = data.get("bleu") if data.get("bleu") is not None else data.get("sacrebleu")
                c = data.get("chrf") if data.get("chrf") is not None else data.get("chrf++")
                m = data.get("meteor")
                cm = data.get("comet")
                return {
                    "bleu": round(float(b), 2) if b is not None else None,
                    "chrf": round(float(c), 2) if c is not None else None,
                    "meteor": round(float(m), 2) if m is not None else None,
                    "comet": round(float(cm), 4) if cm is not None else None
                }
        except Exception:
            pass
    return None

def main():
    root_dir = "checkpoints/tssa_pro"
    report_file = os.path.join(root_dir, "COMPARISON_REPORT.txt")

    lines = []
    lines.append("=" * 85)
    lines.append("📊 BÁO CÁO NGHIỆM THU ĐỐI SOÁT TOÀN DIỆN: TSSA-PRO vs. ALL BASELINES")
    lines.append("=" * 85)
    lines.append("")

    models_config = [
        ("bartpho", "rhade", "tssa_rhade"),
        ("bartpho", "tay", "tssa_tay"),
        ("bartpho", "bahnaric", "tssa_bahnaric"),
        ("vit5", "rhade", "vit5_tssa_rhade"),
        ("vit5", "tay", "vit5_tssa_tay"),
        ("vit5", "bahnaric", "vit5_tssa_bahnaric")
    ]

    table_header = (
        f"{'Model':<10} | {'Lang':<9} | {'BLEU':<6} | {'chrF++':<6} | "
        f"{'Δ vs Vanilla':<14} | {'Δ vs Strongest SOTA':<22} | {'Trạng Thái':<12}"
    )
    lines.append(table_header)
    lines.append("-" * len(table_header))

    all_passed = True

    for backbone, lang, exp_name in models_config:
        ckpt_path = os.path.join(root_dir, exp_name)
        res = extract_metrics_from_ckpt(ckpt_path)

        ref = OFFICIAL_REFS[backbone][lang]
        vanilla_b = ref["vanilla"]["bleu"]
        vanilla_c = ref["vanilla"]["chrf"]
        sota_name = ref["strongest"]["name"]
        sota_b = ref["strongest"]["bleu"]
        sota_c = ref["strongest"]["chrf"]

        if res is not None and res["bleu"] is not None:
            b = res["bleu"]
            c = res["chrf"]
            delta_van = b - vanilla_b
            delta_sota = b - sota_b

            status = "+Vanilla" if delta_van > 0 else "Boundary"
            if delta_sota > 0:
                status = "+SOTA"

            row = (
                f"{backbone:<10} | {lang:<9} | {b:<6.2f} | {c:<6.2f} | "
                f"{delta_van:+6.2f} BLEU   | {delta_sota:+6.2f} vs {sota_name:<11} | {status:<12}"
            )
        else:
            row = f"{backbone:<10} | {lang:<9} | {'N/A':<6} | {'N/A':<6} | {'N/A':<14} | {'N/A':<22} | ⏳ PENDING"
            all_passed = False

        lines.append(row)

    lines.append("-" * len(table_header))
    lines.append("")
    lines.append("Báo cáo đối soát thống kê hoàn tất.")
    lines.append("=" * 85)

    report_content = "\n".join(lines)
    print(report_content)

    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_content + "\n")
    print(f"\n[+] Báo cáo đã được lưu trữ tại: {report_file}")

if __name__ == "__main__":
    main()
