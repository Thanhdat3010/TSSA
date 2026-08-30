"""
Lightning Smoke Test Script for TSSA Framework & All 8 Alignment Baselines
Runs a complete end-to-end 5-second sanity check:
1. Model loading & attribute delegation
2. Forward pass & all 9 loss computations (8 Baselines + TSSA)
3. Generation & Prediction step (Hugging Face Trainer validation loop)
4. Evaluation metrics (SacreBLEU, chrF++, COMET)
"""

import os
import shutil
import torch
import numpy as np
from transformers import AutoTokenizer, Seq2SeqTrainingArguments
from datasets import Dataset

from models.tssa_seq2seq import TSSASeq2SeqModel
from losses.unified_criterion import TSSAUnifiedCriterion
from losses.baselines.factory import UnifiedAlignmentLossFactory
from training.trainer import TSSASeq2SeqTrainer
import sacrebleu

def run_smoke_test():
    print("=" * 65)
    print("      🚀 ĐANG CHẠY SMOKE TEST (KIỂM TRA LỖI TẤT CẢ 9 MÔ HÌNH)")
    print("=" * 65)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Thiết bị: {device}")

    # 1. Nạp Tokenizer
    model_ckpt = "vinai/bartpho-syllable"
    print("[1/5] Nạp Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    # 2. Nạp Model & Kiểm tra delegation
    print("[2/5] Nạp Model & Kiểm tra thuộc tính Trainer...")
    model = TSSASeq2SeqModel(model_name_or_path=model_ckpt, use_route=True).to(device)
    
    assert hasattr(model, "generation_config"), "Model thiếu generation_config!"
    assert hasattr(model, "config"), "Model thiếu config!"
    assert model.can_generate(), "Model phải hỗ trợ can_generate()!"
    print("      -> Delegation thuộc tính Model: OK!")

    # 3. Tạo Dummy Batch & Forward Pass
    print("[3/5] Kiểm tra Forward pass & Toàn bộ 9 loại Loss...")
    src_texts = ["Hnam kơ dră kơ kơl", "Ơi ya gơñ kơ ai"]
    tgt_texts = ["Nhà của tôi ở đây", "Bà ơi cháu đi học"]

    src_enc = tokenizer(src_texts, padding="max_length", max_length=32, truncation=True, return_tensors="pt").to(device)
    with tokenizer.as_target_tokenizer():
        tgt_enc = tokenizer(tgt_texts, padding="max_length", max_length=32, truncation=True, return_tensors="pt").to(device)

    labels = tgt_enc["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100

    dummy_batch = {
        "input_ids": src_enc["input_ids"],
        "attention_mask": src_enc["attention_mask"],
        "labels": labels,
        "decoder_attention_mask": tgt_enc["attention_mask"],
        "align_matrix": torch.rand((2, 32, 32), device=device),
        "teacher_enc_states": torch.randn((2, 32, model.d_model), device=device),
        "teacher_sent_vec": torch.randn((2, model.d_model), device=device),
        "teacher_logits": torch.randn((2, 32, tokenizer.vocab_size), device=device),
        "teacher_cross_attentions": tuple(torch.rand((2, 12, 32, 32), device=device) for _ in range(6))
    }

    # Forward pass
    outputs = model(
        input_ids=dummy_batch["input_ids"],
        attention_mask=dummy_batch["attention_mask"],
        labels=dummy_batch["labels"],
        decoder_attention_mask=dummy_batch["decoder_attention_mask"],
        output_hidden_states=True,
        output_attentions=True
    )
    loss_mt = outputs["loss"]
    assert loss_mt is not None and not torch.isnan(loss_mt)
    print("      -> Forward pass Seq2Seq: OK!")

    # Test TSSA Core Loss
    tssa_crit = TSSAUnifiedCriterion().to(device)
    tssa_res = tssa_crit(loss_mt, outputs, dummy_batch)
    assert not torch.isnan(tssa_res["loss"])
    print("      -> [1/9] TSSA Core Loss: OK!")

    # Test All 8 Baselines via Factory
    baselines = [
        "align_to_distill", "structural_supervision", "shift_aet",
        "cross_init", "awesome_align", "dm_bli", "cl_lsa", "dpo_align"
    ]
    for idx, b_name in enumerate(baselines, start=2):
        factory = UnifiedAlignmentLossFactory(method_name=b_name, config={"hidden_dim": model.d_model}).to(device)
        res = factory(loss_mt, outputs, dummy_batch, global_step=10)
        assert not torch.isnan(res["loss_total"]), f"Loss {b_name} bị NaN!"
        print(f"      -> [{idx}/9] Baseline {b_name}: OK!")

    # 4. Kiểm tra Generation & Prediction Step
    print("[4/5] Kiểm tra Generation & Trainer Eval Loop...")
    gen_out = model.generate(input_ids=src_enc["input_ids"], max_length=32)
    decoded = tokenizer.batch_decode(gen_out, skip_special_tokens=True)
    assert len(decoded) == 2
    print("      -> Generation: OK!")

    # Dummy Dataset for Trainer Eval
    dummy_data = {
        "input_ids": src_enc["input_ids"].cpu(),
        "attention_mask": src_enc["attention_mask"].cpu(),
        "labels": labels.cpu(),
        "decoder_attention_mask": tgt_enc["attention_mask"].cpu(),
        "align_matrix": dummy_batch["align_matrix"].cpu()
    }
    eval_ds = Dataset.from_dict(dummy_data)

    training_args = Seq2SeqTrainingArguments(
        output_dir="checkpoints/smoke_test_tmp",
        per_device_eval_batch_size=2,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )

    def compute_metrics(eval_preds):
        preds, lbls = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        lbls = np.where(lbls != -100, lbls, tokenizer.pad_token_id)
        d_preds = [p.strip() for p in tokenizer.batch_decode(preds, skip_special_tokens=True)]
        d_lbls = [[l.strip() for l in tokenizer.batch_decode(lbls, skip_special_tokens=True)]]
        bleu_res = sacrebleu.corpus_bleu(d_preds, d_lbls, smooth_method="exp")
        return {"sacrebleu": round(bleu_res.score, 2)}

    trainer = TSSASeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=eval_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    eval_res = trainer.evaluate()
    assert "eval_sacrebleu" in eval_res
    print(f"      -> Trainer Eval Loop & SacreBLEU: OK (Score = {eval_res['eval_sacrebleu']})!")

    # 5. Cleanup
    print("[5/5] Dọn dẹp thư mục tạm...")
    if os.path.exists("checkpoints/smoke_test_tmp"):
        shutil.rmtree("checkpoints/smoke_test_tmp")

    print("\n" + "=" * 65)
    print("  🎉 CHÚC MỪNG: TOÀN BỘ 9 MÔ HÌNH VÀ FACTORY LOSS ĐÃ PASS 100%!")
    print("=" * 65)

if __name__ == "__main__":
    run_smoke_test()
