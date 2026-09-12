"""
Inspection Script: Detailed Training Dynamics & Epoch Trajectories for UniTSSA 4.0 & 5.0
Analyzes:
1. Epoch-by-epoch validation SacreBLEU trajectory from trainer_state.json
2. Loss component breakdown (loss_mt, loss_struct, loss_prime, loss_route)
3. Router gate specialization and active head percentage
4. Text run logs for both v4 and v5
"""

import sys
import os
import json
import glob

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def inspect_trainer_states(version_dir, version_name):
    print("=" * 105)
    print(f"       📈 QUỸ ĐẠO HỌC TẬP QUA TỪNG EPOCH CỦA {version_name.upper()} ({version_dir})")
    print("=" * 105)

    if not os.path.exists(version_dir):
        print(f"[-] Không tìm thấy thư mục: {version_dir}")
        return

    subdirs = [d for d in os.listdir(version_dir) if os.path.isdir(os.path.join(version_dir, d)) and d != "ablation_logs"]
    subdirs = sorted(subdirs)

    header = f"{'MÔ HÌNH':<22} | {'EP 1':<8} | {'EP 2':<8} | {'EP 3':<8} | {'EP 4':<8} | {'EP 5':<8} | {'BEST VAL':<9} | {'BEST EP'}"
    print(header)
    print("-" * 105)

    for sd in subdirs:
        state_file = os.path.join(version_dir, sd, "trainer_state.json")
        epoch_bleus = {}
        best_bleu = 0.0
        best_ep = 0

        # 1. Đọc trainer_state.json trực tiếp (đã được fix trong UniTSSA 6.0)
        if os.path.exists(state_file):
            try:
                with open(state_file, "r", encoding="utf-8") as f:
                    state = json.load(f)
                for entry in state.get("log_history", []):
                    if "eval_sacrebleu" in entry:
                        ep = int(round(entry.get("epoch", 0)))
                        b = entry.get("eval_sacrebleu", 0.0)
                        epoch_bleus[ep] = b
                        if b > best_bleu:
                            best_bleu = b
                            best_ep = ep
            except Exception:
                pass

        # 2. Fallback sang ablation_logs/*_training_dynamics.json nếu cần
        if not epoch_bleus:
            dyn_file = os.path.join(version_dir, "ablation_logs", f"{sd}_training_dynamics.json")
            if os.path.exists(dyn_file):
                try:
                    with open(dyn_file, "r", encoding="utf-8") as f:
                        dyn_data = json.load(f)
                    for entry in dyn_data.get("eval_trajectory", []):
                        ep = int(round(entry.get("epoch", 0)))
                        metrics = entry.get("metrics", {})
                        b = metrics.get("eval_sacrebleu", metrics.get("sacrebleu", 0.0))
                        if b:
                            epoch_bleus[ep] = float(b)
                            if float(b) > best_bleu:
                                best_bleu = float(b)
                                best_ep = ep
                except Exception:
                    pass

        if not epoch_bleus:
            print(f"{sd:<22} | -- Chưa có log trajectory (sẽ có trong v6.0) --")
            continue

            ep1_s = f"{epoch_bleus.get(1, '--'):<8}" if isinstance(epoch_bleus.get(1, '--'), str) else f"{epoch_bleus.get(1, 0.0):<8.2f}"
            ep2_s = f"{epoch_bleus.get(2, '--'):<8}" if isinstance(epoch_bleus.get(2, '--'), str) else f"{epoch_bleus.get(2, 0.0):<8.2f}"
            ep3_s = f"{epoch_bleus.get(3, '--'):<8}" if isinstance(epoch_bleus.get(3, '--'), str) else f"{epoch_bleus.get(3, 0.0):<8.2f}"
            ep4_s = f"{epoch_bleus.get(4, '--'):<8}" if isinstance(epoch_bleus.get(4, '--'), str) else f"{epoch_bleus.get(4, 0.0):<8.2f}"
            ep5_s = f"{epoch_bleus.get(5, '--'):<8}" if isinstance(epoch_bleus.get(5, '--'), str) else f"{epoch_bleus.get(5, 0.0):<8.2f}"

            print(f"{sd:<22} | {ep1_s} | {ep2_s} | {ep3_s} | {ep4_s} | {ep5_s} | {best_bleu:<9.2f} | Ep {best_ep}")
        except Exception as e:
            print(f"{sd:<22} | Lỗi phân tích: {e}")

    print("-" * 105)

def inspect_ablation_dynamics(version_dir, version_name):
    log_dir = os.path.join(version_dir, "ablation_logs")
    print("\n" + "=" * 105)
    print(f"       📊 CHI TIẾT LOSS & ROUTER GATES CỦA {version_name.upper()} ({log_dir})")
    print("=" * 105)

    if not os.path.exists(log_dir):
        print(f"[-] Không tìm thấy thư mục ablation_logs: {log_dir}")
        return

    files = sorted(glob.glob(os.path.join(log_dir, "*_training_dynamics.json")))
    if not files:
        print(f"[-] Không có file training_dynamics trong {log_dir}")
        return

    for f_path in files:
        exp_name = os.path.basename(f_path).replace("_training_dynamics.json", "")
        try:
            with open(f_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            steps = data.get("step_history_sample", [])
            gates = data.get("epoch_gates", [])

            first_step = steps[0] if steps else {}
            last_step = steps[-1] if steps else {}

            print(f"\n[+] Thí nghiệm: {exp_name}")
            if first_step and last_step:
                l_start = first_step.get("losses", {})
                l_end = last_step.get("losses", {})
                print(f"    - Step Đầu : loss_mt={l_start.get('loss_mt', 0):.4f}, loss_struct={l_start.get('loss_struct', 0):.4f}, loss_prime={l_start.get('loss_prime', 0):.4f}, total={l_start.get('loss_total', 0):.4f}")
                print(f"    - Step Cuối: loss_mt={l_end.get('loss_mt', 0):.4f}, loss_struct={l_end.get('loss_struct', 0):.4f}, loss_prime={l_end.get('loss_prime', 0):.4f}, total={l_end.get('loss_total', 0):.4f}")

            if gates:
                g_last = gates[-1]
                ep = g_last.get("epoch", "?")
                act = g_last.get("mean_activation", 0.0)
                heads = g_last.get("active_heads", 0)
                tot = g_last.get("total_heads", 0)
                pct = (heads / tot) * 100 if tot > 0 else 0
                print(f"    - Final Router: Epoch {ep}, Mean Gate = {act:.4f}, Active Heads = {heads}/{tot} ({pct:.1f}%)")
        except Exception as e:
            print(f"[!] Lỗi đọc {f_path}: {e}")

    print("=" * 105)

def main():
    # 1. UniTSSA 4.0
    inspect_trainer_states("checkpoints/tssa_v4", "UniTSSA 4.0")
    inspect_ablation_dynamics("checkpoints/tssa_v4", "UniTSSA 4.0")

    # 2. UniTSSA 5.0
    inspect_trainer_states("checkpoints/tssa_v5", "UniTSSA 5.0")
    inspect_ablation_dynamics("checkpoints/tssa_v5", "UniTSSA 5.0")

if __name__ == "__main__":
    main()
