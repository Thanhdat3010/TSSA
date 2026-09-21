#!/usr/bin/env python3
"""
scripts/report_fair_benchmark.py
Fast, zero-latency reporter for Fair Benchmark Suite.
Reads eval_metrics.json directly from checkpoints/fair_benchmark/ and prints
a comprehensive comparison table in < 1 second (NO slow bootstrap resampling).

Outputs:
- SacreBLEU, chrF++, METEOR, COMET
- Delta vs Vanilla Baseline
- Summary markdown saved to checkpoints/fair_benchmark/FAIR_BENCHMARK_REPORT.md
"""

import os
import sys
import json
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

LANGUAGES = [
    {"code": "tay", "name": "Tày (tay -> vi)"},
    {"code": "rhade", "name": "Ê Đê (rhade -> vi)"},
    {"code": "bahnaric", "name": "Ba Na (bahnaric -> vi)"}
]

METHODS = [
    {"id": "vanilla", "name": "Vanilla Baseline", "is_ref": True},
    {"id": "awesome_align", "name": "AWESOME-align (EACL 2021)", "is_ref": False},
    {"id": "cl_lsa", "name": "CL-LSA (NAACL 2021)", "is_ref": False},
    {"id": "align_to_distill", "name": "Align-to-Distill (COLING 2024)", "is_ref": False},
    {"id": "shift_aet", "name": "Shift-AET (EMNLP 2020)", "is_ref": False},
    {"id": "tssa_pro", "name": "TSSA-Pro (Old Anchor)", "is_ref": False},
    {"id": "v4_sent", "name": "TSSA-V4 (Sentence InfoNCE)", "is_ref": False},
    {"id": "v4_tok", "name": "TSSA-V4 (Token Barycenter)", "is_ref": False},
    {"id": "v4_hybrid", "name": "TSSA-V4 (Hybrid Dual-Level)", "is_ref": False}
]

def load_metrics(ckpt_dir):
    metric_file = os.path.join(ckpt_dir, "eval_metrics.json")
    if os.path.exists(metric_file):
        try:
            with open(metric_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                b = data.get("bleu", data.get("sacrebleu"))
                c = data.get("chrf", data.get("chrf++"))
                m = data.get("meteor")
                cm = data.get("comet")
                return {
                    "bleu": float(b) if b is not None else None,
                    "chrf": float(c) if c is not None else None,
                    "meteor": float(m) if (m is not None and m != "--") else None,
                    "comet": float(cm) if cm is not None else None
                }
        except Exception:
            return None
    return None

def format_val(val, digits=2):
    if val is None:
        return "--"
    return f"{val:.{digits}f}"

def format_delta(val):
    if val is None:
        return "--"
    sign = "+" if val >= 0 else ""
    return f"{sign}{val:.2f}"

def main():
    root_dir = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/fair_benchmark"
    report_lines = []

    def p(text=""):
        print(text)
        report_lines.append(text)

    p("=" * 105)
    p(" 📊 BÁO CÁO KẾT QUẢ ĐỐI CHUẨN CÔNG BẰNG (FAIR BENCHMARK REPORT)")
    p(f"    Thư mục lưu trữ: {root_dir}")
    p("=" * 105)

    found_any = False

    for lang_info in LANGUAGES:
        lang_code = lang_info["code"]
        lang_name = lang_info["name"]

        # Kiểm tra xem có thư mục nào của ngôn ngữ này không
        lang_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and f"_{lang_code}_" in d] if os.path.exists(root_dir) else []
        if not lang_dirs:
            continue

        found_any = True
        p("")
        p(f"▶ CẶP NGÔN NGỮ: {lang_name.upper()}")
        p("-" * 105)
        header = f"{'Hệ Thống / Phương Pháp':<28} | {'Seed':<6} | {'SacreBLEU':<10} | {'chrF++':<9} | {'METEOR':<8} | {'COMET':<8} | {'Δ BLEU':<10} | {'Δ chrF++':<10}"
        p(header)
        p("-" * len(header))

        # Tìm Vanilla Baseline làm mốc chuẩn
        vanilla_metrics = None
        for seed in [42, 43, 44]:
            cand = os.path.join(root_dir, f"vanilla_{lang_code}_seed{seed}")
            m = load_metrics(cand)
            if m is not None:
                vanilla_metrics = m
                break

        for m_info in METHODS:
            m_id = m_info["id"]
            m_name = m_info["name"]

            # Quét các seed có sẵn
            for seed in [42, 43, 44]:
                ckpt_path = os.path.join(root_dir, f"{m_id}_{lang_code}_seed{seed}")
                res = load_metrics(ckpt_path)
                if res is None:
                    continue

                b_str = format_val(res["bleu"])
                c_str = format_val(res["chrf"])
                m_str = format_val(res["meteor"])
                cm_str = format_val(res["comet"], digits=4)

                if m_id == "vanilla":
                    d_b_str = "Ref (0.00)"
                    d_c_str = "Ref (0.00)"
                else:
                    if vanilla_metrics is not None and res["bleu"] is not None and vanilla_metrics["bleu"] is not None:
                        d_b = res["bleu"] - vanilla_metrics["bleu"]
                        d_b_str = format_delta(d_b)
                    else:
                        d_b_str = "--"

                    if vanilla_metrics is not None and res["chrf"] is not None and vanilla_metrics["chrf"] is not None:
                        d_c = res["chrf"] - vanilla_metrics["chrf"]
                        d_c_str = format_delta(d_c)
                    else:
                        d_c_str = "--"

                row = f"{m_name:<28} | {seed:<6} | {b_str:<10} | {c_str:<9} | {m_str:<8} | {cm_str:<8} | {d_b_str:<10} | {d_c_str:<10}"
                p(row)

    p("")
    p("=" * 105)
    if not found_any:
        p(f"[*] Chưa tìm thấy checkpoint nào đã hoàn thành trong {root_dir}.")
        p(f"    Vui lòng chạy: bash scripts/run_fair_benchmark.sh tay 42")
    else:
        p("🎉 [HOÀN TẤT BÁO CÁO] Đã xuất số liệu đối chuẩn công bằng siêu tốc!")
    p("=" * 105)

    # Lưu ra file markdown
    if os.path.exists(root_dir):
        report_file = os.path.join(root_dir, "FAIR_BENCHMARK_REPORT.md")
        with open(report_file, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines) + "\n")
        print(f"\n[+] Báo cáo đã được lưu vào: {report_file}")

if __name__ == "__main__":
    main()
