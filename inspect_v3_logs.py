"""
Inspection Script: Detailed Learning Dynamics of TSSA 3.0 & Past Runs
Extracts epoch-by-epoch BLEU scores, training loss, and validation loss
from trainer_state.json across all checkpoints in checkpoints/tssa_v3/
"""

import sys
import os
import json

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def inspect_runs(target_dir="checkpoints/tssa_v3"):
    if not os.path.exists(target_dir):
        print(f"[-] Thư mục {target_dir} không tồn tại.")
        return

    print("=" * 95)
    print(f"       📈 BÁO CÁO TIẾN TRÌNH HỌC TẬP QUA TỪNG EPOCH ({target_dir})")
    print("=" * 95)
    print(f"{'MÔ HÌNH':<22} | {'EP 1':<7} | {'EP 2':<7} | {'EP 3':<7} | {'EP 4':<7} | {'EP 5':<7} | {'BEST BLEU':<10} | {'BEST EP'}")
    print("-" * 95)

    experiments = sorted([d for d in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, d)) and not d.startswith(".")])

    for exp in experiments:
        exp_path = os.path.join(target_dir, exp)
        state_file = os.path.join(exp_path, "trainer_state.json")

        if not os.path.exists(state_file):
            print(f"{exp:<22} | {'-- Không tìm thấy trainer_state.json --':<65}")
            continue

        try:
            with open(state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"{exp:<22} | Lỗi đọc file: {e}")
            continue

        log_history = data.get("log_history", [])
        epoch_bleus = {}
        epoch_losses = {}

        for entry in log_history:
            if "epoch" in entry:
                ep = int(round(entry["epoch"]))
                if "eval_sacrebleu" in entry:
                    epoch_bleus[ep] = entry["eval_sacrebleu"]
                if "eval_loss" in entry:
                    epoch_losses[ep] = entry["eval_loss"]

        ep_strs = []
        for i in range(1, 6):
            if i in epoch_bleus:
                ep_strs.append(f"{epoch_bleus[i]:.2f}")
            else:
                ep_strs.append("--")

        best_metric = data.get("best_metric")
        if best_metric is None and epoch_bleus:
            best_metric = max(epoch_bleus.values())
        best_str = f"{best_metric:.2f}" if best_metric is not None else "--"

        # Tìm best epoch
        best_ep = "--"
        if best_metric is not None:
            for ep, b in epoch_bleus.items():
                if abs(b - best_metric) < 1e-4:
                    best_ep = f"Epoch {ep}"
                    break

        print(f"{exp:<22} | {ep_strs[0]:<7} | {ep_strs[1]:<7} | {ep_strs[2]:<7} | {ep_strs[3]:<7} | {ep_strs[4]:<7} | {best_str:<10} | {best_ep}")

    print("=" * 95)

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/tssa_v3"
    inspect_runs(target)
