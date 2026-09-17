#!/usr/bin/env python3
"""
TSSA Experiment Results Aggregator & LaTeX Exporter
Author: UniTSSA Team
Purpose:
  Extracts all remaining experimental results (Length slicing, Hard instances,
  Bootstrap significance, Qualitative tables, Ablation) and formats them into
  clean terminal tables and copy-paste-ready LaTeX blocks for main.tex.
"""

import os
import sys
import re
import argparse
import pandas as pd

# Support UTF-8 encoding on Windows terminal
if sys.platform == "win32" and sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def parse_markdown_table(md_content, section_keyword):
    """Parses a markdown table following a section header."""
    lines = md_content.splitlines()
    table_lines = []
    found_section = False
    
    for line in lines:
        if section_keyword.lower() in line.lower():
            found_section = True
            continue
        if found_section:
            if "|" in line:
                table_lines.append(line)
            elif line.strip() == "" and len(table_lines) > 0:
                break
                
    if not table_lines:
        return None
        
    header = [c.strip() for c in table_lines[0].split("|")[1:-1]]
    data = []
    for line in table_lines[2:]:  # skip separator line
        if "|" in line:
            row = [c.strip() for c in line.split("|")[1:-1]]
            if len(row) == len(header):
                data.append(row)
                
    if not data:
        return None
    return pd.DataFrame(data, columns=header)

def print_header(title):
    print("\n" + "=" * 90)
    print(f"  [*] {title}")
    print("=" * 90)

def main():
    parser = argparse.ArgumentParser(description="Collect and export all TSSA experimental results")
    parser.add_argument("--docs_dir", type=str, default="docs", help="Path to docs directory")
    parser.add_argument("--latex", action="store_true", help="Print ready-to-paste LaTeX tables")
    args = parser.parse_args()

    docs_dir = args.docs_dir
    bartpho_md = os.path.join(docs_dir, "LENGTH_AND_HARD_ANALYSIS_BARTPHO.md")
    vit5_md = os.path.join(docs_dir, "LENGTH_AND_HARD_ANALYSIS_VIT5.md")
    bartpho_tex = os.path.join(docs_dir, "qualitative_table_bartpho.tex")
    vit5_tex = os.path.join(docs_dir, "qualitative_table_vit5.tex")
    ablation_tex = os.path.join(docs_dir, "ablation_dual_backbone_tay.tex")
    official_md = os.path.join(docs_dir, "OFFICIAL_EXPERIMENT_RESULTS.md")

    print("\n" + "#" * 90)
    print("      [+] TSSA TONG HOP KET QUA THUC NGHIEM (EVAL RESULTS)")
    print("#" * 90)

    # 1. HARD VS EASY INSTANCES
    print_header("1. KẾT QUẢ PHÂN RÃ CÂU KHÓ VS. CÂU DỄ (HARD VS. EASY INSTANCES)")
    found_hard = False
    for name, path in [("BARTpho-syllable", bartpho_md), ("ViT5-base", vit5_md)]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            df = parse_markdown_table(content, "Câu Khó vs. Câu Dễ")
            if df is not None:
                print(f"\n[+] Backbone: {name}")
                print(df.to_string(index=False))
                found_hard = True
    if not found_hard:
        print("[!] Chưa có file LENGTH_AND_HARD_ANALYSIS_*.md. Vui lòng chạy scripts/run_remaining_experiments.sh trước.")

    # 2. SENTENCE LENGTH SLICING
    print_header("2. KẾT QUẢ PHÂN RÃ THEO ĐỘ DÀI CÂU (SENTENCE LENGTH SLICING)")
    found_len = False
    for name, path in [("BARTpho-syllable", bartpho_md), ("ViT5-base", vit5_md)]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            df = parse_markdown_table(content, "Độ Dài Câu")
            if df is not None:
                print(f"\n[+] Backbone: {name}")
                print(df.to_string(index=False))
                found_len = True
    if not found_len:
        print("[!] Chưa có dữ liệu độ dài câu.")

    # 3. SIGNIFICANCE TESTS SUMMARY
    print_header("3. KIỂM ĐỊNH Ý NGHĨA THỐNG KÊ (BOOTSTRAP SIGNIFICANCE P-VALUES)")
    if os.path.exists(official_md):
        with open(official_md, "r", encoding="utf-8") as f:
            content = f.read()
        for sec in ["II.b. Kiểm Định Ý Nghĩa Thống Kê (Paired Bootstrap Resampling - B=1,000)", 
                    "II.c. Kiểm Định Ý Nghĩa Thống Kê Trên ViT5 (Paired Bootstrap Resampling - B=1,000)"]:
            df = parse_markdown_table(content, sec)
            if df is not None:
                print(f"\n[+] Phần: {sec.split('(')[0].strip()}")
                cols = [c for c in df.columns if any(k in c.lower() for k in ["ngôn ngữ", "đối thủ", "chỉ số", "mức tăng", "p-value", "khoảng"])]
                if cols:
                    print(df[cols].to_string(index=False))
                else:
                    print(df.to_string(index=False))

    # 4. QUALITATIVE EXAMPLES
    print_header("4. CAC MAU CAU DICH DINH TINH (QUALITATIVE EXAMPLES)")
    for name, path in [("BARTpho", bartpho_tex), ("ViT5", vit5_tex)]:
        if os.path.exists(path):
            print(f"[v] Da tao san file LaTeX dinh tinh: {path} (kich thuoc: {os.path.getsize(path)} bytes)")
        else:
            print(f"[!] Chua co file: {path}")

    # 5. ABLATION STUDY
    print_header("5. KET QUA ABLATION STUDY TREN CA 2 BACKBONE")
    if os.path.exists(ablation_tex):
        print(f"[v] Da tim thay bang ablation hoan chinh tai: {ablation_tex}")
        with open(ablation_tex, "r", encoding="utf-8") as f:
            print(f.read())
    else:
        print("[*] Ablation study dang cho chay hoac chua hoan tat (file docs/ablation_dual_backbone_tay.tex chua xuat hien).")

    print("\n" + "=" * 90)
    print("  [+] HOAN TAT TRICH XUAT TOAN BO KET QUA THUC NGHIEM!")
    print("=" * 90 + "\n")

if __name__ == "__main__":
    main()
