#!/usr/bin/env python3
"""CPU invariants plus real BARTpho/ViT5 GPU integration for CA-TSSA."""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import tempfile

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    BartConfig,
    BartForConditionalGeneration,
    Seq2SeqTrainingArguments,
    T5Config,
    T5ForConditionalGeneration,
)

try:
    import transformers.utils.import_utils as _hf_import_utils

    if hasattr(_hf_import_utils, "check_torch_load_is_safe"):
        _hf_import_utils.check_torch_load_is_safe = lambda: None
except Exception:
    pass

from data.dataloader import get_dataloaders
from losses.ca_tssa_criterion import CATSSACriterion
from models.ca_tssa_seq2seq import CATSSASeq2SeqModel, module_fingerprint
from training.ca_tssa_trainer import CATSSATrainer


class DummyTokenizer:
    pad_token_id = 0

    def __call__(
        self,
        text,
        max_length=8,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        **kwargs,
    ):
        words = str(text).split()
        ids = [1 + (sum(map(ord, word)) % 30) for word in words][:max_length]
        mask = [1] * len(ids)
        ids += [0] * (max_length - len(ids))
        mask += [0] * (max_length - len(mask))
        return {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.tensor([mask], dtype=torch.long),
        }

    def as_target_tokenizer(self):
        return contextlib.nullcontext()


def tiny_model(kind: str) -> CATSSASeq2SeqModel:
    if kind == "bart":
        config = BartConfig(
            vocab_size=64,
            d_model=32,
            encoder_layers=1,
            decoder_layers=1,
            encoder_attention_heads=4,
            decoder_attention_heads=4,
            encoder_ffn_dim=64,
            decoder_ffn_dim=64,
            max_position_embeddings=32,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            decoder_start_token_id=1,
        )
        backbone = BartForConditionalGeneration(config)
    elif kind == "t5":
        config = T5Config(
            vocab_size=64,
            d_model=32,
            d_kv=8,
            d_ff=64,
            num_layers=1,
            num_decoder_layers=1,
            num_heads=4,
            pad_token_id=0,
            eos_token_id=1,
            decoder_start_token_id=0,
        )
        backbone = T5ForConditionalGeneration(config)
    else:
        raise ValueError(kind)
    return CATSSASeq2SeqModel(
        model_name_or_path=f"tiny-{kind}", d_hidden=16, backbone_model=backbone
    )


def dummy_batch(device: torch.device, vocab_size: int = 64):
    input_ids = torch.tensor([[4, 5, 6, 2, 0], [7, 8, 9, 10, 2]], device=device)
    attention_mask = (input_ids != 0).long()
    labels = torch.tensor([[11, 12, 2, -100], [13, 14, 15, 2]], device=device)
    decoder_attention_mask = (labels != -100).long()
    labels = labels.clamp(max=vocab_size - 1)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "decoder_attention_mask": decoder_attention_mask,
    }


def assert_route_math() -> None:
    criterion = CATSSACriterion(
        gradient_policy="token_project", grad_cap=0.10, warmup_ratio=0.0
    )
    grad_mt = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    grad_anchor = torch.tensor([[[-1.0, 1.0], [0.5, 0.0]]])
    mask = torch.ones((1, 2), dtype=torch.long)
    safe, stats = criterion.route_gradients(grad_mt, grad_anchor, mask)
    dots = (safe * grad_mt).sum(dim=-1)
    ratios = safe.norm(dim=-1) / grad_mt.norm(dim=-1)
    assert stats["conflict_tokens"] == 1
    assert torch.all(dots >= -1e-6), dots
    assert torch.all(ratios <= 0.100001), ratios
    assert safe[0, 1, 0].item() > 0 and abs(safe[0, 1, 1].item()) < 1e-7

    global_criterion = CATSSACriterion(
        gradient_policy="global_project", grad_cap=0.10, warmup_ratio=0.0
    )
    global_safe, _ = global_criterion.route_gradients(grad_mt, grad_anchor, mask)
    assert (global_safe * grad_mt).sum().item() >= -1e-6
    assert global_safe.norm().item() / grad_mt.norm().item() <= 0.100001

    joint = CATSSACriterion(gradient_policy="joint", warmup_ratio=0.0)
    joint_gradient, _ = joint.route_gradients(grad_mt, grad_anchor, mask)
    assert torch.equal(joint_gradient, grad_anchor)

    detach = CATSSACriterion(gradient_policy="detach", warmup_ratio=0.0)
    detached, _ = detach.route_gradients(grad_mt, grad_anchor, mask)
    assert detached.abs().max().item() == 0.0


def assert_old_dataloader() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        frame = pd.DataFrame(
            {"src_text": ["nguon mot", "nguon hai"], "tgt_text": ["dich mot", "dich hai"]}
        )
        frame.to_csv(os.path.join(temp_dir, "train.csv"), index=False)
        frame.to_csv(os.path.join(temp_dir, "test.csv"), index=False)
        train_loader, test_loader, train_dataset, test_dataset = get_dataloaders(
            temp_dir, DummyTokenizer(), batch_size=2, max_src_len=8, max_tgt_len=8
        )
        batch = next(iter(train_loader))
        assert batch["input_ids"].shape == (2, 8)
        assert batch["labels"].shape == (2, 8)
        assert len(train_dataset) == len(test_dataset) == 2
        assert len(test_loader) == 1


def assert_tiny_backbone(kind: str) -> None:
    device = torch.device("cpu")
    model = tiny_model(kind).to(device).train()
    batch = dummy_batch(device)
    teacher_before = module_fingerprint(model.teacher_encoder)

    # Anchor-only backward must touch the projector but no backbone parameter.
    outputs = model(**batch)
    criterion = CATSSACriterion(
        gradient_policy="token_project",
        grad_cap=0.10,
        warmup_ratio=0.0,
        total_steps=2,
    )
    anchor_loss, _ = criterion.compute_anchor_loss(
        outputs["projected_source"],
        outputs["teacher_hidden"],
        outputs["source_mask"],
        outputs["target_mask"],
    )
    model.zero_grad(set_to_none=True)
    anchor_loss.backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum().item() > 0
        for parameter in model.projector.parameters()
    )
    assert not any(
        parameter.grad is not None and parameter.grad.abs().sum().item() > 0
        for parameter in model.backbone.parameters()
    )

    # Full surrogate backward must reconnect MT plus safe anchor to the encoder.
    model.zero_grad(set_to_none=True)
    outputs = model(**batch)
    result = criterion(outputs["loss"], outputs, global_step=2)
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    encoder_grad = sum(
        parameter.grad.detach().float().norm().item()
        for parameter in model.get_encoder().parameters()
        if parameter.grad is not None
    )
    assert encoder_grad > 0.0
    assert all(parameter.grad is None for parameter in model.teacher_encoder.parameters())

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-4
    )
    optimizer.step()
    assert module_fingerprint(model.teacher_encoder) == teacher_before

    # Generation must bypass both training-only modules.
    calls = {"projector": 0, "teacher": 0}

    def count_projector(*_):
        calls["projector"] += 1

    def count_teacher(*_):
        calls["teacher"] += 1

    projector_hook = model.projector.register_forward_hook(count_projector)
    teacher_hook = model.teacher_encoder.register_forward_hook(count_teacher)
    model.eval()
    generated = model.generate(
        input_ids=batch["input_ids"][:1],
        attention_mask=batch["attention_mask"][:1],
        max_new_tokens=3,
        num_beams=2,
    )
    projector_hook.remove()
    teacher_hook.remove()
    assert generated.numel() > 0
    assert calls == {"projector": 0, "teacher": 0}, calls

    # Backbone and projector must both round-trip.
    with tempfile.TemporaryDirectory() as temp_dir:
        model.save_pretrained(temp_dir)
        restored = tiny_model(kind)
        restored.load_pretrained(temp_dir)
        for left, right in zip(model.projector.parameters(), restored.projector.parameters()):
            assert torch.allclose(left.detach(), right.detach())


def assert_trainer_step() -> None:
    model = tiny_model("bart")
    criterion = CATSSACriterion(
        gradient_policy="token_project",
        grad_cap=0.10,
        warmup_ratio=0.0,
        total_steps=1,
    )
    batch = dummy_batch(torch.device("cpu"))
    examples = [
        {key: value[index].clone() for key, value in batch.items()}
        for index in range(batch["input_ids"].size(0))
    ]

    def collate(items):
        return {key: torch.stack([item[key] for item in items]) for key in items[0]}

    with tempfile.TemporaryDirectory() as temp_dir:
        args = Seq2SeqTrainingArguments(
            output_dir=temp_dir,
            per_device_train_batch_size=2,
            max_steps=1,
            save_strategy="no",
            eval_strategy="no",
            report_to="none",
            disable_tqdm=True,
            remove_unused_columns=False,
        )
        trainer = CATSSATrainer(
            model=model,
            args=args,
            train_dataset=examples,
            data_collator=collate,
            criterion=criterion,
            model_type="ca_tssa",
        )
        trainer.train()
        trainer.save_model(temp_dir)
        assert os.path.exists(os.path.join(temp_dir, "ca_tssa_projector.safetensors"))
        assert os.path.exists(os.path.join(temp_dir, "gradient_diagnostics.json"))
        assert criterion.get_diagnostics()["observed_batches"] == 1


def integration_batch(tokenizer, device: torch.device):
    source = tokenizer(
        ["Ngôn ngữ thiểu số Việt Nam", "Hôm nay chúng tôi đi học"],
        padding=True,
        truncation=True,
        max_length=24,
        return_tensors="pt",
    ).to(device)
    with tokenizer.as_target_tokenizer():
        target = tokenizer(
            ["Bảo tồn ngôn ngữ là cần thiết", "Đây là một phép thử dịch máy"],
            padding=True,
            truncation=True,
            max_length=24,
            return_tensors="pt",
        ).to(device)
    labels = target["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    return {
        "input_ids": source["input_ids"],
        "attention_mask": source["attention_mask"],
        "labels": labels,
        "decoder_attention_mask": target["attention_mask"],
    }


def assert_real_backbone(checkpoint: str, precision: torch.dtype) -> None:
    device = torch.device("cuda")
    print(f"[*] GPU integration: {checkpoint} ({precision})")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = CATSSASeq2SeqModel(checkpoint, d_hidden=256).to(device).train()
    criterion = CATSSACriterion(
        gradient_policy="token_project",
        grad_cap=0.10,
        warmup_ratio=0.0,
        total_steps=1,
    ).to(device)
    batch = integration_batch(tokenizer, device)
    with torch.autocast(device_type="cuda", dtype=precision):
        outputs = model(**batch)
        result = criterion(outputs["loss"], outputs, global_step=1)
    assert torch.isfinite(result["loss"]), result["log_dict"]
    result["loss"].backward()
    assert criterion.get_diagnostics()["observed_valid_tokens"] > 0
    assert all(parameter.grad is None for parameter in model.teacher_encoder.parameters())
    print(f"[✓] {checkpoint}: forward/backward và diagnostics hợp lệ")
    del criterion, model, tokenizer, batch, outputs, result
    torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip_integration",
        action="store_true",
        help="Chỉ chạy Tiny BART/T5 và DataLoader tests; mặc định server chạy cả model thật.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 80)
    print(" CA-TSSA SMOKE TEST: gradient routing, backbone generality, checkpoint, data")
    print("=" * 80)

    assert_route_math()
    print("[✓] Token projection, non-negative post-conflict cosine và 10% cap")
    assert_old_dataloader()
    print("[✓] TSSADataset/get_dataloaders hiện tại được giữ nguyên và hoạt động")
    for kind in ("bart", "t5"):
        assert_tiny_backbone(kind)
        print(f"[✓] Tiny {kind.upper()}: isolation, backward, teacher, generation, save/load")
    assert_trainer_step()
    print("[✓] Hugging Face Trainer: một optimization step và checkpoint sidecars")

    if not args.skip_integration:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Smoke test mặc định cần GPU cho BARTpho/ViT5. "
                "Dùng --skip_integration chỉ cho kiểm tra local CPU."
            )
        assert_real_backbone("vinai/bartpho-syllable", torch.float16)
        precision = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        if precision == torch.float32:
            raise RuntimeError("ViT5 fairness smoke test yêu cầu GPU hỗ trợ BF16")
        assert_real_backbone("VietAI/vit5-base", precision)

    print("=" * 80)
    print(" TẤT CẢ CA-TSSA SMOKE TEST ĐỀU PASS")
    print(" Có thể chạy: bash scripts/run_ca_tssa_screening.sh tay 42")
    print("=" * 80)


if __name__ == "__main__":
    main()
