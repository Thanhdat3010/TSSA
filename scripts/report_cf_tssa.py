#!/usr/bin/env python3
"""Validate completed CF/Vanilla runs and report H or F without mixing protocols."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

ROOT = "checkpoints/fair_benchmark"
BACKBONES = {"bartpho": ("vinai/bartpho-syllable", 2e-5),
             "vit5": ("VietAI/vit5-base", 1e-4)}


def run_name(method, backbone, lang, seed, protocol):
    return f"{method}_cf_{backbone}_{lang}_seed{seed}_{protocol}"


def load_run(output_dir, method, backbone, lang, seed, protocol):
    names = []
    if method == "vanilla" and backbone == "bartpho" and protocol == "fixed_final":
        names.append(f"vanilla_{lang}_seed{seed}_fixed_final")
    names.append(run_name(method, backbone, lang, seed, protocol))
    error = None
    for name in names:
        directory = os.path.join(output_dir, name)
        try:
            return validated_metrics(directory, method, backbone, lang, seed, protocol), directory
        except (ValueError, OSError, json.JSONDecodeError) as caught:
            error = caught
    raise error


def validated_metrics(directory, method, backbone, lang, seed, protocol):
    required = ["eval_metrics.json", "test_predictions.csv", "config.json"]
    if method == "cf_tssa":
        required += ["cf_tssa_config.json", "cf_tssa_diagnostics.json"]
    for name in required:
        if not os.path.isfile(os.path.join(directory, name)):
            raise ValueError(f"Missing {directory}/{name}")
    if not any(os.path.isfile(os.path.join(directory, name))
               for name in ("model.safetensors", "pytorch_model.bin")):
        raise ValueError(f"Missing model weights in {directory}")
    with open(os.path.join(directory, "eval_metrics.json"), encoding="utf-8") as stream:
        metrics = json.load(stream)
    meta = metrics.get("metadata", {})
    checkpoint, lr = BACKBONES[backbone]
    expected = dict(model_type="cf_tssa" if method == "cf_tssa" else "vanilla",
                    model_ckpt=checkpoint, lang=lang, seed=seed,
                    learning_rate=lr, num_epochs=5, batch_size=16,
                    weight_decay=0.01, warmup_steps=500,
                    max_source_length=256, max_target_length=256,
                    num_beams=4, length_penalty=1.0, selection_protocol=protocol,
                    fp16=backbone == "bartpho", bf16=backbone == "vit5")
    if method == "cf_tssa":
        expected.update(cf_tau=0.10, cf_eta=0.10, cf_margin=1e-4,
                        cf_grad_budget=0.05, cf_warmup_ratio=0.10,
                        trainable_extra_parameters=0)
        diag = metrics.get("cf_diagnostics", {})
        if not diag or diag.get("observed_batches", 0) == 0:
            raise ValueError(f"Missing CF diagnostics: {directory}")
    for key, value in expected.items():
        actual = meta.get(key)
        if isinstance(value, float):
            valid = isinstance(actual, (float, int)) and abs(actual - value) < 1e-9
        else:
            valid = actual == value
        if not valid:
            raise ValueError(f"{directory}: {key}={actual!r}, expected {value!r}")
    if "sacrebleu" not in metrics:
        raise ValueError(f"Missing SacreBLEU: {directory}")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("check", "report"))
    parser.add_argument("--protocol", required=True, choices=("legacy_best", "fixed_final"))
    parser.add_argument("--method", choices=("vanilla", "cf_tssa"))
    parser.add_argument("--backbone", choices=tuple(BACKBONES))
    parser.add_argument("--lang", choices=("tay", "rhade", "bahnaric"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output_dir", default=ROOT)
    args = parser.parse_args()
    if args.action == "check":
        if not all((args.method, args.backbone, args.lang, args.seed is not None)):
            parser.error("check requires --method --backbone --lang --seed")
        try:
            _, directory = load_run(args.output_dir, args.method, args.backbone,
                                    args.lang, args.seed, args.protocol)
        except (ValueError, OSError, json.JSONDecodeError) as error:
            print(f"[NOT-COMPLETE] {error}", file=sys.stderr)
            raise SystemExit(1) from None
        print(f"[VALID] {directory}")
        return

    cells = ([ ("bartpho", "tay", 42) ] if args.protocol == "legacy_best" else
             [(backbone, lang, seed) for backbone in BACKBONES
              for lang in ("tay", "rhade", "bahnaric") for seed in (42, 43, 44)])
    lines = [f"# CF-TSSA {args.protocol} report", "",
             "Do not compare BLEU across protocols; legacy_best selects checkpoints on test.", "",
             "| Backbone | Dataset | Seed | Vanilla BLEU | CF BLEU | Delta | CF accepted rate |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    deltas = []
    cell_deltas = {}
    for backbone, lang, seed in cells:
        scores = {}
        for method in ("vanilla", "cf_tssa"):
            scores[method], _ = load_run(args.output_dir, method, backbone, lang, seed,
                                         args.protocol)
        delta = scores["cf_tssa"]["sacrebleu"] - scores["vanilla"]["sacrebleu"]
        deltas.append(delta)
        cell_deltas.setdefault((backbone, lang), []).append(delta)
        accepted = scores["cf_tssa"]["cf_diagnostics"]["accepted_rate"]
        lines.append(f"| {backbone} | {lang} | {seed} | {scores['vanilla']['sacrebleu']:.2f} "
                     f"| {scores['cf_tssa']['sacrebleu']:.2f} | {delta:+.2f} | {accepted:.3f} |")
    mean = statistics.mean(deltas)
    lines += ["", f"Mean paired delta BLEU: **{mean:+.2f}**."]
    if args.protocol == "legacy_best":
        lines.append("Historical exploratory gate: GO to fixed-final only if delta > 0 and pilot passed; no hyperparameter selection.")
        lines.append("Decision: **" + ("GO for user review" if mean > 0 else "NO-GO") + "**.")
    else:
        means = {key: statistics.mean(values) for key, values in cell_deltas.items()}
        backbone_means = {backbone: statistics.mean(value for (name, _), value in means.items()
                                                  if name == backbone) for backbone in BACKBONES}
        positive_cells = sum(value > 0 for value in means.values())
        success = (mean >= .30 and all(value > 0 for value in backbone_means.values())
                   and positive_cells >= 4 and all(value >= -.20 for value in means.values()))
        lines.append(f"Positive cells: {positive_cells}/6; backbone means: {backbone_means}.")
        lines.append("Numerical gate: **" + ("PASS" if success else "FAIL") +
                     "**. Novelty and ablations remain separate requirements.")
    filename = ("CF_TSSA_HISTORICAL_REPORT.md" if args.protocol == "legacy_best" else
                "CF_TSSA_FIXED_FINAL_REPORT.md")
    path = os.path.join(args.output_dir, filename)
    with open(path, "w", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
