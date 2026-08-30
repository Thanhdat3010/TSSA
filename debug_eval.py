"""
Diagnostic and Debug Script for NMT Evaluation (TSSA)
Tests test_predictions.csv across all metric formats, displays sample translations,
and prints full n-gram statistics for SacreBLEU, chrF++, METEOR, and COMET.
"""

import os
import sys
import argparse
import pandas as pd

def debug_predictions(csv_path: str):
    print("=" * 70)
    print(f"🔍 ĐANG CHẨN ĐOÁN CHI TIẾT FILE BẢN DỊCH: {csv_path}")
    print("=" * 70)

    if not os.path.exists(csv_path):
        print(f"[!] Lỗi: Không tìm thấy file tại {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"[+] Tổng số câu trong file: {len(df)}")
    print(f"[+] Các cột trong file: {list(df.columns)}")

    # 1. Kiểm tra mẫu dữ liệu
    print("\n--- 5 MẪU DỊCH ĐẦU TIÊN ---")
    for i in range(min(5, len(df))):
        src = df.iloc[i].get("source", "")
        ref = df.iloc[i].get("reference", "")
        pred = df.iloc[i].get("prediction", "")
        print(f"[{i+1}]")
        print(f"  SRC : {src}")
        print(f"  REF : {ref}")
        print(f"  PRED: {pred}")

    sources = df["source"].astype(str).fillna("").tolist()
    references = df["reference"].astype(str).fillna("").tolist()
    predictions = df["prediction"].astype(str).fillna("").tolist()

    # 2. Kiểm tra SacreBLEU với các định dạng đầu vào khác nhau
    print("\n" + "=" * 70)
    print("📊 KIỂM TRA ĐỊNH DẠNG SACREBLEU:")
    print("=" * 70)

    try:
        import sacrebleu
        
        # Định dạng Chuẩn SacreBLEU: [references] (Shape: 1 x N)
        bleu_standard = sacrebleu.corpus_bleu(predictions, [references])
        print(f"\n[1] Chuẩn SacreBLEU (Format [references], No Smoothing):")
        print(f"    -> Score: {bleu_standard.score:.2f}")
        print(f"    -> Chi tiết n-grams: {bleu_standard.precisions}")
        print(f"    -> Brevity Penalty (BP): {bleu_standard.bp:.4f} (sys_len={bleu_standard.sys_len}, ref_len={bleu_standard.ref_len})")

        bleu_exp = sacrebleu.corpus_bleu(predictions, [references], smooth_method="exp")
        print(f"\n[2] Chuẩn SacreBLEU (Format [references], Smooth 'exp'):")
        print(f"    -> Score: {bleu_exp.score:.2f}")
        print(f"    -> Chi tiết n-grams: {bleu_exp.precisions}")

        # Định dạng Lỗi cũ: [[r] for r in references] (Shape: N x 1)
        try:
            bleu_old = sacrebleu.corpus_bleu(predictions, [[r] for r in references])
            print(f"\n[3] Định dạng cũ ([[r] for r in references]):")
            print(f"    -> Score: {bleu_old.score:.2f}")
        except Exception as e:
            print(f"\n[3] Định dạng cũ bị lỗi: {e}")

        # chrF++
        chrf_res = sacrebleu.corpus_chrf(predictions, [references], word_order=2)
        print(f"\n[4] chrF++: {chrf_res.score:.2f}")

    except Exception as e:
        print(f"[!] Lỗi khi chạy SacreBLEU: {e}")

    # 3. METEOR
    try:
        import nltk
        from nltk.translate.meteor_score import meteor_score
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
        m_scores = [meteor_score([r.split()], p.split()) for r, p in zip(references, predictions)]
        meteor_val = round((sum(m_scores) / len(m_scores)) * 100, 2)
        print(f"\n[5] METEOR: {meteor_val}")
    except Exception as e:
        print(f"[!] Lỗi khi tính METEOR: {e}")

    # 4. COMET
    try:
        from comet import load_from_checkpoint, download_model
        model_path = download_model("Unbabel/wmt22-comet-da")
        comet_metric = load_from_checkpoint(model_path)
        comet_data = [{"src": s, "mt": p, "ref": r} for s, p, r in zip(sources, predictions, references)]
        comet_output = comet_metric.predict(comet_data, batch_size=32, gpus=1 if "cuda" in str(sys.argv) or True else 0)
        print(f"\n[6] COMET: {comet_output.system_score:.4f}")
    except Exception as e:
        print(f"[!] COMET test: {e}")

    print("\n" + "=" * 70)
    print("🏁 HOÀN TẤT CHẨN ĐOÁN!")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="checkpoints/bartpho_vanilla_tay/test_predictions.csv")
    args = parser.parse_args()
    debug_predictions(args.csv)
