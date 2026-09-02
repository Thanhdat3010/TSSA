"""
Automated Results & Checkpoint Inspector for TSSA 2.0 & Baseline Benchmarks
Computes all 4 standard evaluation metrics: BLEU, chrF++, METEOR, and COMET.
Scans both checkpoints/ (5 epochs) and checkpoints_10epochs/ (10 epochs) directories.
"""

import os
import glob
import json
import argparse
import pandas as pd
import sacrebleu

# NLTK WordNet for METEOR
try:
    import nltk
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    from nltk.translate.meteor_score import meteor_score
    NLTK_AVAILABLE = True
except Exception:
    NLTK_AVAILABLE = False

COMET_MODEL = None

def get_comet_model():
    global COMET_MODEL
    if COMET_MODEL is None:
        try:
            from comet import download_model, load_from_checkpoint
            print("[*] Đang nạp mô hình COMET (Unbabel/wmt20-comet-da)...")
            model_path = download_model("Unbabel/wmt20-comet-da")
            COMET_MODEL = load_from_checkpoint(model_path)
        except Exception as e:
            print(f"[!] Không thể nạp mô hình COMET: {e}")
            COMET_MODEL = False
    return COMET_MODEL if COMET_MODEL is not False else None

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

def compute_metrics_for_pred_file(pred_file, compute_comet=False):
    metrics_cache_file = os.path.join(os.path.dirname(pred_file), "eval_metrics.json")
    cached_metrics = {}
    if os.path.exists(metrics_cache_file):
        try:
            with open(metrics_cache_file, "r", encoding="utf-8") as f:
                cached_metrics = json.load(f)
        except Exception:
            pass

    if cached_metrics and "bleu" in cached_metrics and "chrf" in cached_metrics and "meteor" in cached_metrics:
        if not compute_comet or ("comet" in cached_metrics and cached_metrics["comet"] != "--"):
            return cached_metrics

    df = pd.read_csv(pred_file)
    num_samples = len(df)
    
    src_col, ref_col, pred_col = None, None, None
    for col in df.columns:
        col_lower = col.lower()
        if "src" in col_lower or "source" in col_lower:
            src_col = col
        elif "ref" in col_lower or "target" in col_lower:
            ref_col = col
        elif "pred" in col_lower or "translation" in col_lower or "hyp" in col_lower:
            pred_col = col

    if not (ref_col and pred_col):
        return None

    refs = [str(t).strip() for t in df[ref_col].fillna("").tolist()]
    preds = [str(p).strip() for p in df[pred_col].fillna("").tolist()]
    srcs = [str(s).strip() for s in df[src_col].fillna("").tolist()] if src_col else [""] * len(preds)

    # 1. SacreBLEU
    b_res = sacrebleu.corpus_bleu(preds, [refs], smooth_method="exp")
    bleu_score = round(b_res.score, 2)

    # 2. chrF++
    c_res = sacrebleu.corpus_chrf(preds, [refs], word_order=2)
    chrf_score = round(c_res.score, 2)

    # 3. METEOR
    meteor_val = cached_metrics.get("meteor", "--")
    if meteor_val == "--" and NLTK_AVAILABLE:
        try:
            m_scores = [meteor_score([r.split()], p.split()) for r, p in zip(refs, preds)]
            meteor_val = round((sum(m_scores) / max(1, len(m_scores))) * 100, 2)
        except Exception:
            meteor_val = "--"

    # 4. COMET
    comet_val = cached_metrics.get("comet", "--")
    if compute_comet and (comet_val == "--" or comet_val is None):
        c_model = get_comet_model()
        if c_model is not None and src_col:
            try:
                comet_data = [{"src": s, "mt": p, "ref": r} for s, p, r in zip(srcs, preds, refs)]
                out = c_model.predict(comet_data, batch_size=64, gpus=1)
                comet_val = round(out.system_score, 4)
            except Exception as e:
                print(f"[!] Lỗi tính COMET cho {pred_file}: {e}")
                comet_val = "--"

    res = {
        "num_samples": num_samples,
        "bleu": bleu_score,
        "chrf": chrf_score,
        "meteor": meteor_val,
        "comet": comet_val
    }

    try:
        with open(metrics_cache_file, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

    return res

def summarize_checkpoints_in_dir(target_dir, compute_comet=False):
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

        pred_file = os.path.join(cdir, "test_predictions.csv")
        m_dict = None
        if os.path.exists(pred_file):
            try:
                m_dict = compute_metrics_for_pred_file(pred_file, compute_comet=compute_comet)
            except Exception:
                pass

        results.append({
            "Mục": target_dir,
            "Mô Hình / Experiment": folder_name,
            "Dung Lượng": dir_size,
            "Trọng Số": weight_status,
            "Test": m_dict["num_samples"] if m_dict else "-",
            "BLEU ↑": f"{m_dict['bleu']:.2f}" if m_dict and m_dict["bleu"] is not None else "-",
            "chrF++ ↑": f"{m_dict['chrf']:.2f}" if m_dict and m_dict["chrf"] is not None else "-",
            "METEOR ↑": f"{m_dict['meteor']}" if m_dict else "-",
            "COMET ↑": f"{m_dict['comet']}" if m_dict else "-"
        })

    return results

def main():
    parser = argparse.ArgumentParser(description="Summarize evaluation results across all models.")
    parser.add_argument("--comet", action="store_true", default=False, help="Tính thêm điểm COMET (Unbabel/wmt20-comet-da)")
    args = parser.parse_args()

    print("=" * 130)
    print("      📊 BÁO CÁO TOÀN DIỆN 4 CHỈ SỐ CHUẨN (BLEU, chrF++, METEOR, COMET)")
    print("=" * 130)

    all_results = []
    if os.path.exists("checkpoints"):
        all_results.extend(summarize_checkpoints_in_dir("checkpoints", compute_comet=args.comet))
    if os.path.exists("checkpoints_10epochs"):
        all_results.extend(summarize_checkpoints_in_dir("checkpoints_10epochs", compute_comet=args.comet))

    if not all_results:
        print("[!] Chưa tìm thấy checkpoint nào.")
        return

    df_res = pd.DataFrame(all_results)
    print(df_res.to_markdown(index=False))
    print("=" * 130)
    if not args.comet:
        print("💡 Gợi ý: Gõ 'python summary_results.py --comet' để tính thêm chỉ số COMET trên GPU!")
        print("=" * 130)

if __name__ == "__main__":
    main()
