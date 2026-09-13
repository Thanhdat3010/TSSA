"""
Diagnostic Tool for UniTSSA Final Runs
Reads trainer_state.json and test_predictions.csv across all 6 models in checkpoints/tssa_final/
Outputs:
1. Epoch-by-epoch validation progression (loss & BLEU)
2. Brevity Penalty (BP), System Length vs Reference Length ratio
3. Individual n-gram precisions (p1, p2, p3, p4) and chrF++
"""

import os
import glob
import json
import pandas as pd

try:
    import sacrebleu
except ImportError:
    sacrebleu = None

def diagnose():
    base_dir = os.path.join("checkpoints", "tssa_final")
    if not os.path.exists(base_dir):
        print(f"[!] Thư mục {base_dir} không tồn tại!")
        return

    models = [
        ("BARTpho Rhade", "tssa_rhade"),
        ("BARTpho Tay", "tssa_tay"),
        ("BARTpho Bahnar", "tssa_bahnaric"),
        ("ViT5 Rhade", "vit5_tssa_rhade"),
        ("ViT5 Tay", "vit5_tssa_tay"),
        ("ViT5 Bahnar", "vit5_tssa_bahnaric")
    ]

    print("=" * 110)
    print("       🔍 BÁO CÁO CHẨN ĐOÁN CHI TIẾT TIẾN TRÌNH & N-GRAM CỦA UniTSSA FINAL (6 MÔ HÌNH)")
    print("=" * 110)

    for label, folder in models:
        ckpt_path = os.path.join(base_dir, folder)
        print(f"\n>>> [{label.upper()}] ({folder})")
        if not os.path.exists(ckpt_path):
            print(f"    [!] Chưa tìm thấy thư mục: {ckpt_path}")
            continue

        # 1. Đọc trainer_state.json
        state_file = os.path.join(ckpt_path, "trainer_state.json")
        if os.path.exists(state_file):
            try:
                with open(state_file, "r", encoding="utf-8") as f:
                    state = json.load(f)
                best_m = state.get("best_metric", "N/A")
                best_ckpt = state.get("best_model_checkpoint", "N/A")
                print(f"    [*] Best Val BLEU : {best_m} (tại: {os.path.basename(str(best_ckpt))})")
                print("    [*] Lịch sử Validation qua các Epoch:")
                history = state.get("log_history", [])
                eval_entries = [h for h in history if "eval_sacrebleu" in h or "eval_loss" in h]
                for h in eval_entries:
                    ep = h.get("epoch", "?")
                    loss = h.get("eval_loss", "--")
                    bleu = h.get("eval_sacrebleu", "--")
                    loss_str = f"{loss:.4f}" if isinstance(loss, float) else str(loss)
                    bleu_str = f"{bleu:.2f}" if isinstance(bleu, float) else str(bleu)
                    print(f"        - Epoch {ep}: Val Loss = {loss_str} | Val BLEU = {bleu_str}")
            except Exception as e:
                print(f"    [!] Lỗi khi đọc trainer_state.json: {e}")
        else:
            print("    [*] Không tìm thấy trainer_state.json")

        # 2. Đọc test_predictions.csv và phân tích n-gram + Brevity Penalty
        pred_file = os.path.join(ckpt_path, "test_predictions.csv")
        if os.path.exists(pred_file) and sacrebleu is not None:
            try:
                df = pd.read_csv(pred_file)
                ref_col, pred_col = None, None
                for c in df.columns:
                    cl = str(c).lower().strip()
                    if "ref" in cl or "target" in cl:
                        ref_col = c
                    elif "pred" in cl or "hyp" in cl:
                        pred_col = c

                if ref_col and pred_col:
                    preds = [str(p).strip() if pd.notna(p) else "" for p in df[pred_col].tolist()]
                    refs = [str(r).strip() if pd.notna(r) else "" for r in df[ref_col].tolist()]

                    b_exp = sacrebleu.corpus_bleu(preds, [refs], smooth_method="exp")
                    b_std = sacrebleu.corpus_bleu(preds, [refs])
                    c_val = sacrebleu.corpus_chrf(preds, [refs], word_order=2).score

                    ratio = b_std.sys_len / max(1, b_std.ref_len)
                    p1, p2, p3, p4 = b_std.precisions

                    print(f"    [*] Kết quả Test Set Chi Tiết:")
                    print(f"        - SacreBLEU (exp) : {b_exp.score:.2f} | chrF++: {c_val:.2f}")
                    print(f"        - Brevity Penalty : {b_std.bp:.4f} (SysLen={b_std.sys_len}, RefLen={b_std.ref_len}, Tỷ lệ={ratio:.3f})")
                    print(f"        - n-gram Precision: 1-gram={p1:.1f}% | 2-gram={p2:.1f}% | 3-gram={p3:.1f}% | 4-gram={p4:.1f}%")
            except Exception as e:
                print(f"    [!] Lỗi khi phân tích test_predictions.csv: {e}")
        else:
            print("    [*] Không tìm thấy test_predictions.csv hoặc chưa cài sacrebleu")

    print("\n" + "=" * 110)

if __name__ == "__main__":
    diagnose()
