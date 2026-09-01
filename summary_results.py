"""
Automated Results & Checkpoint Inspector for TSSA 2.0 & Baseline Benchmarks
Scans both checkpoints/ (5 epochs) and checkpoints_10epochs/ (10 epochs) directories.
"""

import os
import glob
import pandas as pd
import sacrebleu

def get_dir_size_str(path):
    total_bytes = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_bytes += os.path.getsize(fp)
    if total_bytes >= 1024 * 1024 * 1024:
        return f"{total_bytes / (1024**3):.2f} GB"
    elif total_bytes >= 1024 * 1024:
        return f"{total_bytes / (1024**2):.1f} MB"
    elif total_bytes >= 1024:
        return f"{total_bytes / 1024:.1f} KB"
    return f"{total_bytes} B"

def summarize_checkpoints_in_dir(target_dir):
    checkpoint_dirs = sorted(glob.glob(f"{target_dir}/*"))
    if not checkpoint_dirs:
        return []

    results = []

    for cdir in checkpoint_dirs:
        if not os.path.isdir(cdir):
            continue
        folder_name = os.path.basename(cdir)
        dir_size = get_dir_size_str(cdir)
        
        all_files = os.listdir(cdir)
        has_safetensors = "model.safetensors" in all_files
        has_bin = "pytorch_model.bin" in all_files
        sub_ckpts = [f for f in all_files if os.path.isdir(os.path.join(cdir, f)) and f.startswith("checkpoint-")]

        if has_safetensors:
            weight_status = "✅ model.safetensors"
        elif has_bin:
            weight_status = "✅ pytorch_model.bin"
        elif len(sub_ckpts) > 0:
            weight_status = f"📁 {len(sub_ckpts)} sub-checkpoints"
        else:
            weight_status = "📄 Chỉ có file kết quả"

        bleu_score = None
        chrf_score = None
        num_samples = None

        pred_file = os.path.join(cdir, "test_predictions.csv")
        if os.path.exists(pred_file):
            try:
                df = pd.read_csv(pred_file)
                num_samples = len(df)
                
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
            except Exception:
                pass

        results.append({
            "Mục Lưu Trữ": target_dir,
            "Mô Hình / Experiment": folder_name,
            "Dung Lượng": dir_size,
            "Trọng Số": weight_status,
            "Số Mẫu Test": num_samples if num_samples is not None else "-",
            "BLEU": f"{bleu_score:.2f}" if bleu_score is not None else "-",
            "chrF++": f"{chrf_score:.2f}" if chrf_score is not None else "-"
        })

    return results

def main():
    print("=" * 110)
    print("      📊 BÁO CÁO TOÀN DIỆN CHECKPOINTS (5 EPOCHS vs 10 EPOCHS)")
    print("=" * 110)

    all_results = []
    if os.path.exists("checkpoints"):
        all_results.extend(summarize_checkpoints_in_dir("checkpoints"))
    if os.path.exists("checkpoints_10epochs"):
        all_results.extend(summarize_checkpoints_in_dir("checkpoints_10epochs"))

    if not all_results:
        print("[!] Chưa tìm thấy checkpoint nào.")
        return

    df_res = pd.DataFrame(all_results)
    print(df_res.to_markdown(index=False))
    print("=" * 110)

if __name__ == "__main__":
    main()
