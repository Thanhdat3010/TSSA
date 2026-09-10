"""
Inspection Script: Detailed Learning Dynamics of TSSA 3.0 & Past Runs
Extracts step-level losses, epoch gates, and training dynamics from:
1. checkpoints/tssa_v3/ablation_logs/*_training_dynamics.json
2. train_bahnar.log & other *.log files
"""

import sys
import os
import json
import glob
import re

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def inspect_ablation_logs(log_dir="checkpoints/tssa_v3/ablation_logs"):
    print("=" * 95)
    print(f"       📊 CHI TIẾT LOG ĐỘNG HỌC TRAINING DYNAMICS ({log_dir})")
    print("=" * 95)

    if not os.path.exists(log_dir):
        print(f"[-] Không tìm thấy thư mục: {log_dir}")
        return

    files = glob.glob(os.path.join(log_dir, "*_training_dynamics.json"))
    if not files:
        print(f"[-] Không có file *_training_dynamics.json trong {log_dir}")
        print(f"    Các file hiện có: {os.listdir(log_dir)}")
        return

    for f_path in sorted(files):
        exp_name = os.path.basename(f_path).replace("_training_dynamics.json", "")
        try:
            with open(f_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            steps = data.get("step_history_sample", [])
            gates = data.get("epoch_gates", [])
            
            first_step = steps[0] if steps else {}
            last_step = steps[-1] if steps else {}
            
            print(f"\n[+] Thí nghiệm: {exp_name}")
            print(f"    - Tổng số sample steps: {len(steps)}")
            if first_step and last_step:
                print(f"    - Step 0 Loss: {first_step.get('losses', {})}")
                print(f"    - Final Loss : {last_step.get('losses', {})}")
            
            if gates:
                print("    - Tỷ lệ Router Gate qua các Epoch:")
                for g in gates:
                    ep = g.get("epoch", "?")
                    act = g.get("mean_activation", 0.0)
                    heads = g.get("active_heads", 0)
                    tot = g.get("total_heads", 0)
                    print(f"      * Epoch {ep}: Mean Gate = {act:.4f}, Active Heads = {heads}/{tot} ({(heads/tot)*100:.1f}%)")
        except Exception as e:
            print(f"[!] Lỗi đọc {f_path}: {e}")

    print("\n" + "=" * 95)

def inspect_text_logs():
    print("\n" + "=" * 95)
    print("       📜 PHÂN TÍCH FILE TEXT LOGS (*.log)")
    print("=" * 95)
    
    log_files = glob.glob("*.log")
    for lf in log_files:
        size_kb = os.path.getsize(lf) / 1024.0
        print(f"[+] File: {lf} ({size_kb:.1f} KB)")
        
        # Extract sacrebleu and eval lines
        eval_lines = []
        try:
            with open(lf, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if any(kw in line.lower() for kw in ["sacrebleu", "eval_loss", "epoch", "hoàn tất", "kết quả"]):
                        if len(line.strip()) > 0:
                            eval_lines.append(line.strip())
        except Exception as e:
            print(f"    Lỗi đọc file: {e}")
            continue

        if eval_lines:
            print(f"    -> Tìm thấy {len(eval_lines)} dòng liên quan đến đánh giá:")
            for l in eval_lines[-15:]:  # in 15 dòng cuối
                print(f"       {l}")
        else:
            print("    -> Không tìm thấy dòng eval rõ ràng.")
    print("=" * 95)

if __name__ == "__main__":
    inspect_ablation_logs()
    inspect_text_logs()
