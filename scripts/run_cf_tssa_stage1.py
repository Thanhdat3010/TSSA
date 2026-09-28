#!/usr/bin/env python3
"""Train-only 500-step CF pilot: never opens test.csv or computes test metrics."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
from transformers import AutoTokenizer, Seq2SeqTrainingArguments, set_seed

from data.dataloader import TSSADataset, tssa_collate_fn
from losses.cf_tssa_criterion import CFTSSACriterion
from models.cf_tssa_seq2seq import CFTSSASeq2SeqModel
from models.tssa_seq2seq import TSSASeq2SeqModel
from models.tssa_vit5 import TSSAViT5Model
from training.cf_tssa_trainer import CFTSSATrainer
from training.trainer import TSSASeq2SeqTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="tay", choices=("tay", "rhade", "bahnaric"))
    parser.add_argument("--model_ckpt", default="vinai/bartpho-syllable")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--data_dir", default="data_processed")
    parser.add_argument("--output_dir", default="checkpoints/fair_benchmark/cf_tssa_pilot_tay_seed42")
    args = parser.parse_args()
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise FileExistsError(f"Pilot directory is not empty; refusing overwrite: {args.output_dir}")
    train_path = os.path.join(args.data_dir, args.lang, "train.csv")
    if not os.path.isfile(train_path):
        raise FileNotFoundError(train_path)
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_ckpt)
    dataset = TSSADataset(train_path, tokenizer, max_src_len=256, max_tgt_len=256)
    is_t5 = "t5" in args.model_ckpt.lower()
    bf16 = is_t5 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    fp16 = not is_t5 and torch.cuda.is_available()
    model = CFTSSASeq2SeqModel(args.model_ckpt, special_token_ids=tokenizer.all_special_ids)
    criterion = CFTSSACriterion(total_steps=args.steps)
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir, max_steps=args.steps, per_device_train_batch_size=16,
        learning_rate=1e-4 if is_t5 else 2e-5, weight_decay=0.01, warmup_steps=500,
        logging_steps=50, save_strategy="no", eval_strategy="no",
        fp16=fp16, bf16=bf16, seed=args.seed, data_seed=args.seed,
        report_to="none", remove_unused_columns=False,
    )
    trainer = CFTSSATrainer(model=model, args=training_args, train_dataset=dataset,
                            data_collator=tssa_collate_fn, tokenizer=tokenizer,
                            criterion=criterion, model_type="cf_tssa")
    started = time.perf_counter()
    trainer.train()
    seconds = time.perf_counter() - started
    cf_peak_bytes = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    diagnostics = criterion.get_diagnostics()
    if diagnostics["accepted"] == 0:
        raise RuntimeError("Pilot NO-GO: no anchors accepted")

    # Same 500 steps, seed, optimizer and train data for the compute-cost control.
    del trainer, model, criterion
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    set_seed(args.seed)
    vanilla = (TSSAViT5Model(args.model_ckpt, use_route=False) if is_t5 else
               TSSASeq2SeqModel(args.model_ckpt, use_route=False))
    vanilla_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(args.output_dir, "vanilla_timing"), max_steps=args.steps,
        per_device_train_batch_size=16, learning_rate=1e-4 if is_t5 else 2e-5,
        weight_decay=0.01, warmup_steps=500, logging_steps=50,
        save_strategy="no", eval_strategy="no", fp16=fp16, bf16=bf16,
        seed=args.seed, data_seed=args.seed, report_to="none", remove_unused_columns=False,
    )
    vanilla_trainer = TSSASeq2SeqTrainer(
        model=vanilla, args=vanilla_args, train_dataset=dataset,
        data_collator=tssa_collate_fn, tokenizer=tokenizer, model_type="vanilla")
    vanilla_started = time.perf_counter()
    vanilla_trainer.train()
    vanilla_seconds = time.perf_counter() - vanilla_started
    os.makedirs(args.output_dir, exist_ok=True)
    report = {"protocol": "train_only_pilot", "test_data_read": False,
              "model_ckpt": args.model_ckpt, "lang": args.lang, "seed": args.seed,
              "steps_per_method": args.steps, "cf_seconds": seconds,
              "vanilla_seconds": vanilla_seconds,
              "cf_to_vanilla_time_ratio": seconds / vanilla_seconds,
              "cf_seconds_per_step": seconds / args.steps,
              "peak_cf_cuda_bytes": cf_peak_bytes,
              "peak_vanilla_cuda_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
              "diagnostics": diagnostics}
    report["pilot_gate_pass"] = (
        diagnostics["accepted"] > 0 and
        diagnostics["mean_true_uplift"] > diagnostics["mean_control_uplift"] and
        report["cf_to_vanilla_time_ratio"] <= 3)
    with open(os.path.join(args.output_dir, "pilot_report.json"), "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))
    if not report["pilot_gate_pass"]:
        raise RuntimeError("Pilot NO-GO: inspect accepted rate, true/control uplift and cost ratio")


if __name__ == "__main__":
    main()
