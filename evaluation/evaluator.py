"""
Translation Evaluator Module for TSSA
Computes SacreBLEU, chrF++, METEOR, and COMET scores on test datasets.
"""

import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import sacrebleu

class TranslationEvaluator:
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu", use_comet: bool = True):
        self.device = device
        self.use_comet = use_comet

        self.comet_metric = None
        if use_comet:
            try:
                from comet import load_from_checkpoint, download_model
                model_path = download_model("Unbabel/wmt22-comet-da")
                self.comet_metric = load_from_checkpoint(model_path)
                print("[+] Đã nạp COMET model: Unbabel/wmt22-comet-da")
            except Exception as e:
                print(f"[!] Warning: Không thể nạp COMET ({e}). Sẽ bỏ qua COMET.")

    @torch.no_grad()
    def evaluate_model(self, model, tokenizer, dataloader, max_target_len: int = 256, output_save_path: str = None) -> dict:
        """
        Runs generation across dataloader and computes full benchmark metrics.
        """
        model.eval()
        predictions = []
        references = []
        sources = []

        print(f"[*] Đang tiến hành dịch và đánh giá trên {len(dataloader.dataset)} mẫu...")
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_target_len,
                num_beams=4,
                early_stopping=True
            )

            decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            preds_clean = [p.strip() for p in decoded_preds]
            
            # Ground truth targets
            if "tgt_texts" in batch:
                refs_clean = [str(r).strip() for r in batch["tgt_texts"]]
            else:
                labels = batch["labels"].clone()
                labels[labels == -100] = tokenizer.pad_token_id
                refs_clean = [r.strip() for r in tokenizer.batch_decode(labels, skip_special_tokens=True)]

            predictions.extend(preds_clean)
            references.extend(refs_clean)
            if "src_texts" in batch:
                sources.extend([str(s).strip() for s in batch["src_texts"]])

        return self.compute_scores_from_texts(sources, references, predictions, output_save_path=output_save_path)

    def compute_scores_from_texts(self, sources: list, references: list, predictions: list, output_save_path: str = None) -> dict:
        """
        Computes SacreBLEU, chrF++, METEOR, and COMET from text lists.
        """
        # 1. SacreBLEU
        ref_lists = [[r] for r in references]
        bleu_res = sacrebleu.corpus_bleu(predictions, ref_lists)
        bleu_score = bleu_res.score

        # 2. chrF++
        chrf_res = sacrebleu.corpus_chrf(predictions, ref_lists, word_order=2)
        chrf_score = chrf_res.score

        # 3. METEOR calculation
        meteor_val = "--"
        try:
            import nltk
            try:
                nltk.data.find('corpora/wordnet')
            except LookupError:
                nltk.download('wordnet', quiet=True)
                nltk.download('omw-1.4', quiet=True)
            from nltk.translate.meteor_score import meteor_score
            m_scores = [meteor_score([r.split()], p.split()) for r, p in zip(references, predictions)]
            meteor_val = round((sum(m_scores) / len(m_scores)) * 100, 2)
        except Exception as e:
            pass

        # 4. COMET
        comet_score = None
        if self.comet_metric is not None and len(sources) == len(predictions):
            try:
                comet_data = [{"src": s, "mt": p, "ref": r} for s, p, r in zip(sources, predictions, references)]
                comet_output = self.comet_metric.predict(comet_data, batch_size=32, gpus=1 if str(self.device).startswith("cuda") else 0)
                comet_score = comet_output.system_score
            except Exception as e:
                print(f"[!] Lỗi khi tính COMET: {e}")

        results = {
            "sacrebleu": round(bleu_score, 2),
            "chrf++": round(chrf_score, 2),
            "meteor": meteor_val,
            "comet": round(comet_score, 4) if comet_score is not None else "--"
        }

        print(f"\n================ KẾT QUẢ ĐÁNH GIÁ ================")
        print(f"  BLEU   : {results['sacrebleu']}")
        print(f"  chrF++ : {results['chrf++']}")
        print(f"  METEOR : {results['meteor']}")
        print(f"  COMET  : {results['comet']}")
        print(f"==================================================")

        # Lưu bản dịch ra file csv để kiểm tra định tính
        if output_save_path is not None:
            os.makedirs(os.path.dirname(output_save_path), exist_ok=True)
            df_out = pd.DataFrame({
                "source": sources if len(sources) == len(predictions) else [""] * len(predictions),
                "reference": references,
                "prediction": predictions
            })
            df_out.to_csv(output_save_path, index=False, encoding="utf-8")
            print(f"[+] Đã lưu bản dịch kiểm thử vào: {output_save_path}")

        return results
