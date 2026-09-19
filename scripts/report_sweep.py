#!/usr/bin/env python3
"""
scripts/report_sweep.py
Reads all sweep checkpoints in checkpoints/tssa_sweep_tay/ and prints the full benchmark table.
"""

import os
import json
import glob

OFFICIAL_REFS = {
    "vit5": {"vanilla": 34.99, "sota": 35.83, "sota_name": "CL-LSA"},
    "bartpho": {"vanilla": 24.67, "sota": 25.20, "sota_name": "AWESOME-align"}
}

def print_sweep_report(root_dir="checkpoints/tssa_sweep_tay"):
    print("=" * 88)
    print(" 📊 BẢNG TỔNG HỢP TOÀN BỘ KẾT QUẢ LAMBDA-SWEEP TRÊN TIẾNG TÀY (ViT5 & BARTpho)")
    print("=" * 88)
    header = f"{'Model':<10} | {'Lambda':<8} | {'BLEU':<7} | {'chrF++':<7} | {'METEOR':<7} | {'COMET':<8} | {'vs Vanilla':<12} | {'vs SOTA':<12}"
    print(header)
    print("-" * len(header))

    found = False
    for backbone in ["vit5", "bartpho"]:
        ref = OFFICIAL_REFS[backbone]
        pattern = os.path.join(root_dir, f"{backbone}_tay_lambda_*")
        ckpt_dirs = sorted(glob.glob(pattern))

        for d in ckpt_dirs:
            folder_name = os.path.basename(d)
            lambda_val = folder_name.split("_lambda_")[-1]
            metrics_path = os.path.join(d, "eval_metrics.json")
            if os.path.exists(metrics_path):
                found = True
                with open(metrics_path, "r", encoding="utf-8") as f:
                    m = json.load(f)
                b = m.get("bleu", m.get("sacrebleu", 0.0))
                c = m.get("chrf", m.get("chrf++", 0.0))
                met = m.get("meteor", 0.0)
                com = m.get("comet", 0.0)

                dv = b - ref["vanilla"]
                ds = b - ref["sota"]
                sv = f"{'+' if dv >= 0 else ''}{dv:.2f}"
                ss = f"{'+' if ds >= 0 else ''}{ds:.2f}"

                print(f"{backbone.upper():<10} | {lambda_val:<8} | {b:<7.2f} | {c:<7.2f} | {met:<7.2f} | {com:<8.4f} | {sv:<12} | {ss:<12}")

    if not found:
        print(f"[!] Chưa tìm thấy kết quả nào trong {root_dir}")
    print("=" * 88)

if __name__ == "__main__":
    import sys
    r_dir = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/tssa_sweep_tay"
    print_sweep_report(r_dir)
