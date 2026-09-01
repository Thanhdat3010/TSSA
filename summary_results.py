"""
Automated Results & Checkpoint Inspector for TSSA 2.0 & Baseline Benchmarks
Scans checkpoints directory, details all physical files/weights/subfolders, and prints a comprehensive audit.
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

def summarize_all_checkpoints():
    print("=" * 105)
    print("      📊 BÁO CÁO TOÀN DIỆN TIẾN ĐỘ, KẾT QUẢ & DUNG LƯỢNG CHECKPOINTS")
    print("=" * 105)

    checkpoint_dirs = sorted(glob.glob("checkpoints/*"))
    if not checkpoint_dirs:
        print("[!] Chưa tìm thấy checkpoint nào trong thư mục checkpoints/.")
        return

    results = []

    for cdir in checkpoint_dirs:
        if not os.path.isdir(cdir):
            continue
        folder_name = os.path.basename(cdir)
        dir_size = get_dir_size_str(cdir)
        
        # Check files inside
        all_files = os.listdir(cdir)
        has_safetensors = "model.safetensors" in all_files
        has_bin = "pytorch_model.bin" in all_files
        has_test_pred = "test_predictions.csv" in all_files
        sub_ckpts = [f for f in all_files if os.path.isdir(os.path.join(cdir, f)) and f.startswith("checkpoint-")]

        # Determine weight status description
        if has_safetensors:
            weight_status = "✅ model.safetensors"
        elif has_bin:
            weight_status = "✅ pytorch_model.bin"
        elif len(sub_ckpts) > 0:
            weight_status = f"📁 {len(sub_ckpts)} sub-checkpoints"
        else:
            weight_status = "📄 Chỉ có file kết quả (Đã dọn file weights)"

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
            "Mô Hình / Experiment": folder_name,
            "Dung Lượng Thư Mục": dir_size,
            "Trạng Thái Trọng Số": weight_status,
            "Số Mẫu Test": num_samples if num_samples is not None else "-",
            "BLEU (SacreBLEU)": f"{bleu_score:.2f}" if bleu_score is not None else "-",
            "chrF++": f"{chrf_score:.2f}" if chrf_score is not None else "-"
        })

    df_res = pd.DataFrame(results)
    print(df_res.to_markdown(index=False))
    print("=" * 105)

if __name__ == "__main__":
    summarize_all_checkpoints()
