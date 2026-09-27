#!/usr/bin/env python3
"""Pair fixed-final Vanilla and CA-TSSA results by seed and report the gate."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


SEEDS = (42, 43, 44)


def read_result(root: Path, name: str, lang: str, seed: int, method: str) -> dict:
    experiment_dir = root / name
    metrics_path = experiment_dir / "eval_metrics.json"
    prediction_path = experiment_dir / "test_predictions.csv"
    if not metrics_path.is_file() or not prediction_path.is_file():
        raise FileNotFoundError(f"Incomplete experiment: {experiment_dir}")

    with metrics_path.open(encoding="utf-8") as handle:
        result = json.load(handle)
    metadata = result.get("metadata", {})
    expected = {
        "model_type": method,
        "model_ckpt": "vinai/bartpho-syllable",
        "lang": lang,
        "seed": seed,
        "learning_rate": 2e-5,
        "num_epochs": 5,
        "batch_size": 16,
        "weight_decay": 0.01,
        "warmup_steps": 500,
        "max_source_length": 256,
        "max_target_length": 256,
        "fp16": True,
        "bf16": False,
        "num_beams": 4,
        "length_penalty": 1.0,
        "selection_protocol": "fixed_final",
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"{metrics_path}: metadata {key}={metadata.get(key)!r}, expected {value!r}"
            )
    if method == "ca_tssa":
        if metadata.get("gradient_policy") != "token_project":
            raise ValueError(f"{metrics_path}: CA-TSSA is not token_project")
        if not (experiment_dir / "gradient_diagnostics.json").is_file():
            raise FileNotFoundError(f"Missing diagnostics: {experiment_dir}")

    if not isinstance(result.get("sacrebleu"), (int, float)):
        raise ValueError(f"{metrics_path}: missing numeric sacrebleu")
    return result


def format_metric(value: object, digits: int = 2) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return "--"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", choices=("tay", "rhade", "bahnaric"), default="tay")
    parser.add_argument("--output_dir", type=Path, default=Path("checkpoints/fair_benchmark"))
    args = parser.parse_args()

    rows = []
    bleu_deltas = []
    for seed in SEEDS:
        vanilla = read_result(
            args.output_dir,
            f"vanilla_{args.lang}_seed{seed}_fixed_final",
            args.lang,
            seed,
            "vanilla",
        )
        ca_tssa = read_result(
            args.output_dir,
            f"ca_tssa_token_{args.lang}_seed{seed}_fixed_final",
            args.lang,
            seed,
            "ca_tssa",
        )
        delta_bleu = ca_tssa["sacrebleu"] - vanilla["sacrebleu"]
        bleu_deltas.append(delta_bleu)
        rows.append((seed, vanilla, ca_tssa, delta_bleu))

    lines = [
        f"# CA-TSSA fixed-final confirmation: {args.lang} -> vi",
        "",
        "Both methods: BARTpho, 5 fixed epochs, seeds 42/43/44. "
        "Test is evaluated after training, not used to select a checkpoint.",
        "",
        "| Seed | Vanilla BLEU | CA-TSSA BLEU | Delta BLEU | "
        "Vanilla chrF++ | CA-TSSA chrF++ | Vanilla METEOR | CA-TSSA METEOR | "
        "Vanilla COMET | CA-TSSA COMET |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for seed, vanilla, ca_tssa, delta in rows:
        lines.append(
            f"| {seed} | {format_metric(vanilla.get('sacrebleu'))} | "
            f"{format_metric(ca_tssa.get('sacrebleu'))} | {delta:+.2f} | "
            f"{format_metric(vanilla.get('chrf++'))} | "
            f"{format_metric(ca_tssa.get('chrf++'))} | "
            f"{format_metric(vanilla.get('meteor'))} | "
            f"{format_metric(ca_tssa.get('meteor'))} | "
            f"{format_metric(vanilla.get('comet'), 4)} | "
            f"{format_metric(ca_tssa.get('comet'), 4)} |"
        )

    mean_delta = statistics.mean(bleu_deltas)
    positive_seeds = sum(delta > 0 for delta in bleu_deltas)
    gate_passed = mean_delta >= 0.30 and positive_seeds >= 2
    lines.extend(
        [
            "",
            f"Mean paired Delta BLEU: **{mean_delta:+.2f}** "
            f"(sample SD {statistics.stdev(bleu_deltas):.2f}; "
            f"positive seeds {positive_seeds}/3).",
            "",
            "Decision gate: "
            + ("**GO to full benchmark**" if gate_passed else "**STOP before full benchmark**")
            + " (mean Delta BLEU >= +0.30 and at least 2/3 positive seeds).",
            "This gate does not establish statistical significance; "
            "paired bootstrap and the other datasets/backbones remain pending.",
            "",
        ]
    )

    report_path = args.output_dir / f"CA_TSSA_FIXED_FINAL_{args.lang.upper()}_REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
