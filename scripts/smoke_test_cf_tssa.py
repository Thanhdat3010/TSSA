#!/usr/bin/env python3
"""CPU invariants and optional GPU integration for the fixed CF-TSSA method."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
from transformers import (AutoTokenizer, BartConfig, BartForConditionalGeneration,
                          T5Config, T5ForConditionalGeneration)

from losses.cf_tssa_criterion import CFTSSACriterion
from models.ca_tssa_seq2seq import module_fingerprint
from models.cf_tssa_seq2seq import CFTSSASeq2SeqModel
from scripts.smoke_test_ca_tssa import assert_old_dataloader, dummy_batch, integration_batch


def tiny_model(kind):
    if kind == "bart":
        config = BartConfig(vocab_size=64, d_model=32, encoder_layers=1,
                            decoder_layers=1, encoder_attention_heads=4,
                            decoder_attention_heads=4, encoder_ffn_dim=64,
                            decoder_ffn_dim=64, max_position_embeddings=32,
                            pad_token_id=0, bos_token_id=1, eos_token_id=2,
                            decoder_start_token_id=1)
        backbone = BartForConditionalGeneration(config)
    else:
        config = T5Config(vocab_size=64, d_model=32, d_kv=8, d_ff=64,
                          num_layers=1, num_decoder_layers=1, num_heads=4,
                          pad_token_id=0, eos_token_id=2, decoder_start_token_id=0)
        backbone = T5ForConditionalGeneration(config)
    return CFTSSASeq2SeqModel(f"tiny-{kind}", backbone, special_token_ids=(0, 1, 2))


def assert_backbone(model, batch):
    model.train()
    before = module_fingerprint(model.teacher_encoder)
    assert not any(p.requires_grad for p in model.teacher_encoder.parameters())
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == sum(
        p.numel() for p in model.backbone.parameters() if p.requires_grad)
    outputs = model(**batch)
    assert not outputs["source_mask"][0, 3]  # EOS
    assert not outputs["source_mask"][0, 4]  # PAD
    criterion = CFTSSACriterion(total_steps=2, warmup_ratio=0)
    candidates = criterion.build_candidates(outputs)
    assert len(candidates) <= batch["input_ids"].size(0)
    for item in candidates:
        assert torch.allclose(item["true_patch"].norm(), item["wrong_patch"].norm(), rtol=1e-5)
    result = criterion(outputs["loss"], model, outputs, batch, global_step=1)
    assert torch.isfinite(result["loss"])
    assert model.training and model.backbone.model.training and not model.teacher_encoder.training
    if candidates:
        # Deterministic accepted/rejected fixtures exercise the 5% cap and gate.
        forced = CFTSSACriterion(total_steps=1, warmup_ratio=0)
        forced.probe = lambda _model, _outputs, _inputs, _candidates: (
            torch.ones(batch["input_ids"].size(0)), torch.zeros(batch["input_ids"].size(0)))
        accepted_result = forced(outputs["loss"], model, outputs, batch, global_step=1)
        assert forced.get_diagnostics()["accepted"] == len(candidates)
        assert 0 <= accepted_result["log_dict"]["cf_ratio"] <= 0.050001
        rejected = CFTSSACriterion(total_steps=1, warmup_ratio=0)
        rejected.probe = lambda _model, _outputs, _inputs, _candidates: (
            torch.zeros(batch["input_ids"].size(0)), torch.ones(batch["input_ids"].size(0)))
        rejected_result = rejected(outputs["loss"], model, outputs, batch, global_step=1)
        assert rejected_result["loss"] is outputs["loss"]
        result = accepted_result
    result["loss"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.get_encoder().parameters())
    assert all(p.grad is None for p in model.teacher_encoder.parameters())
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    optimizer.step()
    assert module_fingerprint(model.teacher_encoder) == before
    assert criterion.get_diagnostics()["observed_batches"] == 1

    calls = {"teacher": 0}
    hook = model.teacher_encoder.register_forward_hook(
        lambda *_: calls.__setitem__("teacher", calls["teacher"] + 1))
    model.eval()
    model.generate(batch["input_ids"][:1], attention_mask=batch["attention_mask"][:1],
                   max_new_tokens=2, num_beams=2)
    hook.remove()
    assert calls["teacher"] == 0
    with tempfile.TemporaryDirectory() as directory:
        model.save_pretrained(directory)
        assert os.path.exists(os.path.join(directory, "cf_tssa_config.json"))
        criterion.save_diagnostics(directory)
        assert os.path.exists(os.path.join(directory, "cf_tssa_diagnostics.json"))
        restored = CFTSSASeq2SeqModel("restored", model.backbone.model.__class__(model.config),
                                     special_token_ids=(0, 1, 2))
        restored.load_pretrained(directory)
        assert all(torch.allclose(a, b) for a, b in zip(model.backbone.parameters(),
                                                         restored.backbone.parameters()))

    model.train()
    single = {key: value[:1] for key, value in batch.items()}
    one_output = model(**single)
    one_result = criterion(one_output["loss"], model, one_output, single, global_step=1)
    assert one_result["loss"] is one_output["loss"]  # no cross-sentence control


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_integration", action="store_true")
    args = parser.parse_args()
    assert_old_dataloader()
    print("[PASS] Existing TSSA DataLoader")
    for kind in ("bart", "t5"):
        model = tiny_model(kind)
        assert_backbone(model, dummy_batch(torch.device("cpu")))
        print(f"[PASS] Tiny {kind.upper()}: gate, gradients, teacher, save/load, generation")
    if not args.skip_integration:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Full smoke test requires BF16-capable CUDA GPU")
        for checkpoint, dtype in (("vinai/bartpho-syllable", torch.float16),
                                  ("VietAI/vit5-base", torch.bfloat16)):
            tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            model = CFTSSASeq2SeqModel(checkpoint, special_token_ids=tokenizer.all_special_ids).cuda()
            batch = integration_batch(tokenizer, torch.device("cuda"))
            with torch.autocast("cuda", dtype=dtype):
                outputs = model.train()(**batch)
                result = CFTSSACriterion(total_steps=1)(outputs["loss"], model, outputs,
                                                         batch, global_step=0)
            assert torch.isfinite(result["loss"])
            result["loss"].backward()
            assert all(p.grad is None for p in model.teacher_encoder.parameters())
            print(f"[PASS] {checkpoint}: {dtype}, forward/backward")
            del model, tokenizer, outputs, result, batch
            torch.cuda.empty_cache()
    print("ALL CF-TSSA SMOKE TESTS PASS")


if __name__ == "__main__":
    main()
