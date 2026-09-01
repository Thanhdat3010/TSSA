"""
Automated Results & Progress Logger for TSSA 2.0 & Baseline Benchmarks
Scans checkpoints directory, detects all finished experiments, and prints a formatted summary table.
"""

import os
import glob
import json
import pandas as pd
import sacrebleu

def summarize_all_checkpoints():
    print("=" * 95)
    print("      📊 BÁO CÁO TIẾN ĐỘ & KẾT QUẢ CÁC MÔ HÌNH ĐÃ HUẤN LUYỆN (CHECKPOINTS)")
    print("=" * 95)

    checkpoint_dirs = sorted(glob.glob("checkpoints/*"))
    if not checkpoint_dirs:
        print("[!] Chưa tìm thấy checkpoint nào trong thư mục checkpoints/.")
        return

    results = []

    for cdir in checkpoint_dirs:
        if not os.path.isdir(cdir):
            continue
        folder_name = os.path.basename(cdir)
        
        # Check if test_predictions.csv exists
        pred_file = os.path.join(cdir, "test_predictions.csv")
        has_weights = os.path.exists(os.path.join(cdir, "model.safetensors")) or os.path.exists(os.path.join(cdir, "pytorch_model.bin"))
        
        bleu_score = None
        chrf_score = None
        num_samples = None

        if os.path.exists(pred_file):
            try:
                df = pd.read_csv(pred_file)
                num_samples = len(df)
                
                # Column name resolution
                ref_col = None
                pred_col = None
                for col in df.columns:
                    col_lower = col.lower()
                    if "ref" in col_lower or "target" in col_lower:
                        ref_col = col
                    elif "pred" in col_lower or "translation" in col_lower or "hyp" in col_lower:
                        pred_col = col

                if ref_col and pred_col:
                    refs = [[str(t).strip() for t in df[ref_col].fillna("").tolist()]]
                    preds = [str(p).strip() for p in df[pred_col].fillna("").tolist()]
                    
                    b_res = sacrebleu.corpus_bleu(preds, refs, smooth_method="exp")
                    c_res = sacrebleu.corpus_chrf(preds, refs, word_order=2)
                    bleu_score = b_res.score
                    chrf_score = c_res.score
            except Exception as e:
                pass

        results.append({
            "Mô Hình / Experiment": folder_name,
            "Trạng Thái": "✅ Hoàn tất" if (pred_file and bleu_score is not None) else "⏳ Đang chạy/Chưa xong",
            "Số Mẫu Test": num_samples if num_samples is not None else "-",
            "BLEU (SacreBLEU)": f"{bleu_score:.2f}" if bleu_score is not None else "-",
            "chrF++": f"{chrf_score:.2f}" if chrf_score is not None else "-",
            "Trọng Số (Weights)": "✅ Đang lưu" if has_weights else "🗑️ Đã giải phóng"
        })

    df_res = pd.DataFrame(results)
    print(df_res.to_markdown(index=False))
    print("=" * 95)

if __name__ == "__main__":
    summarize_all_checkpoints()
