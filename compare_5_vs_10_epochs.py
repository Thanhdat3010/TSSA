"""
Comparative Evaluator: 5 Epochs vs 10 Epochs
Compares SacreBLEU and chrF++ across 3 low-resource languages between 5-epoch and 10-epoch models.
"""

import os
import glob
import pandas as pd
import sacrebleu

def evaluate_pred_file(pred_file):
    if not os.path.exists(pred_file):
        return None, None
    try:
        df = pd.read_csv(pred_file)
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
            return round(b_res.score, 2), round(c_res.score, 2)
    except Exception:
        pass
    return None, None

def main():
    print("=" * 110)
    print("      📊 BẢNG SO SÁNH HIỆU NĂNG HUẤN LUYỆN: 5 EPOCHS VS 10 EPOCHS (VANILLA vs TSSA 2.0)")
    print("=" * 110)

    langs = [("rhade", "Ê Đê"), ("tay", "Tày"), ("bahnaric", "Ba Na")]
    models = [("bartpho_vanilla", "Vanilla BARTpho"), ("tssa", "TSSA 2.0 (Ours)")]

    table_data = []

    for lang_code, lang_name in langs:
        for model_code, model_name in models:
            exp_id = f"{model_code}_{lang_code}"
            
            # 5 Epochs
            pred_5 = os.path.join("checkpoints", exp_id, "test_predictions.csv")
            b5, c5 = evaluate_pred_file(pred_5)
            
            # 10 Epochs
            pred_10 = os.path.join("checkpoints_10epochs", exp_id, "test_predictions.csv")
            b10, c10 = evaluate_pred_file(pred_10)

            # Compute deltas
            if b5 is not None and b10 is not None:
                delta_b = b10 - b5
                delta_b_str = f"{delta_b:+.2f}"
            else:
                delta_b_str = "-"

            if c5 is not None and c10 is not None:
                delta_c = c10 - c5
                delta_c_str = f"{delta_c:+.2f}"
            else:
                delta_c_str = "-"

            table_data.append({
                "Ngôn Ngữ": f"{lang_name} ({lang_code})",
                "Mô Hình": model_name,
                "BLEU (5 Ep)": f"{b5:.2f}" if b5 is not None else "-",
                "BLEU (10 Ep)": f"{b10:.2f}" if b10 is not None else "⏳ Đang chạy",
                "Δ BLEU": delta_b_str,
                "chrF++ (5 Ep)": f"{c5:.2f}" if c5 is not None else "-",
                "chrF++ (10 Ep)": f"{c10:.2f}" if c10 is not None else "⏳ Đang chạy",
                "Δ chrF++": delta_c_str
            })

    df = pd.DataFrame(table_data)
    print(df.to_markdown(index=False))
    print("=" * 110)

if __name__ == "__main__":
    main()
