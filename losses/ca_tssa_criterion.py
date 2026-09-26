"""Semantic barycenter loss and encoder-interface gradient routing for CA-TSSA."""

from __future__ import annotations

import json
import os
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CATSSACriterion(nn.Module):
    POLICIES = {"token_project", "global_project", "joint", "detach"}

    def __init__(
        self,
        gradient_policy: str = "token_project",
        grad_cap: float = 0.10,
        warmup_ratio: float = 0.10,
        total_steps: int = 1,
        tau_align: float = 0.10,
        tau_h: float = 0.50,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if gradient_policy not in self.POLICIES:
            raise ValueError(
                f"Unknown CA-TSSA gradient policy {gradient_policy!r}; "
                f"expected one of {sorted(self.POLICIES)}"
            )
        if grad_cap < 0:
            raise ValueError("grad_cap must be non-negative")
        if not 0 <= warmup_ratio <= 1:
            raise ValueError("warmup_ratio must be in [0, 1]")

        self.gradient_policy = gradient_policy
        self.grad_cap = float(grad_cap)
        self.warmup_ratio = float(warmup_ratio)
        self.total_steps = max(1, int(total_steps))
        self.warmup_steps = max(1, int(round(self.total_steps * self.warmup_ratio)))
        self.tau_align = float(tau_align)
        self.tau_h = float(tau_h)
        self.eps = float(eps)
        self.reset_diagnostics()

    def reset_diagnostics(self) -> None:
        self._diagnostics = {
            "batches": 0,
            "valid_tokens": 0,
            "conflict_tokens": 0,
            "cosine_before_sum": 0.0,
            "cosine_after_sum": 0.0,
            "safe_to_mt_ratio_sum": 0.0,
            "anchor_loss_sum": 0.0,
            "alignment_entropy_sum": 0.0,
            "anchor_weight_sum": 0.0,
            "source_length_sum": 0.0,
            "ramp_sum": 0.0,
        }

    def compute_anchor_loss(
        self,
        projected_source: torch.Tensor,
        teacher_hidden: torch.Tensor,
        source_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        source = F.normalize(projected_source.float(), dim=-1, eps=self.eps)
        target = F.normalize(teacher_hidden.detach().float(), dim=-1, eps=self.eps)

        similarities = torch.einsum("bsd,btd->bst", source, target) / self.tau_align
        valid_target = target_mask.bool().unsqueeze(1)
        similarities = similarities.masked_fill(~valid_target, -1e4)
        posterior = torch.softmax(similarities, dim=-1).detach()

        barycenter = F.normalize(
            torch.einsum("bst,btd->bsd", posterior, target),
            dim=-1,
            eps=self.eps,
        )
        cosine = (source * barycenter).sum(dim=-1).clamp(-1.0, 1.0)

        target_length = target_mask.sum(dim=-1, keepdim=True).clamp(min=2).float()
        entropy = -(posterior * posterior.clamp_min(self.eps).log()).sum(dim=-1)
        normalized_entropy = (entropy / target_length.log()).clamp(0.0, 1.0)
        confidence = torch.exp(-normalized_entropy / self.tau_h)

        valid_source = source_mask.bool()
        denominator = valid_source.float().sum().clamp(min=1.0)
        anchor_loss = (
            confidence * (1.0 - cosine) * valid_source.float()
        ).sum() / denominator
        diagnostics = {
            "mean_anchor_weight": float(confidence[valid_source].mean().detach().cpu())
            if valid_source.any()
            else 0.0,
            "mean_alignment_entropy": float(
                normalized_entropy[valid_source].mean().detach().cpu()
            )
            if valid_source.any()
            else 0.0,
        }
        return anchor_loss, diagnostics

    def _token_project(
        self,
        grad_mt: torch.Tensor,
        grad_anchor: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        dot = (grad_mt * grad_anchor).sum(dim=-1, keepdim=True)
        mt_sq = grad_mt.square().sum(dim=-1, keepdim=True)
        mt_norm = mt_sq.sqrt()
        valid = valid_mask.unsqueeze(-1) & (mt_norm > self.eps)
        conflict = valid & (dot < 0)
        projected = torch.where(
            conflict,
            grad_anchor - (dot / (mt_sq + self.eps)) * grad_mt,
            grad_anchor,
        )
        projected_norm = projected.square().sum(dim=-1, keepdim=True).sqrt()
        scale = torch.clamp(
            self.grad_cap * mt_norm / (projected_norm + self.eps), max=1.0
        )
        return torch.where(valid, projected * scale, torch.zeros_like(projected))

    def _global_project(
        self,
        grad_mt: torch.Tensor,
        grad_anchor: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        expanded_mask = valid_mask.unsqueeze(-1).to(grad_mt.dtype)
        grad_mt = grad_mt * expanded_mask
        grad_anchor = grad_anchor * expanded_mask
        mt_sq = grad_mt.square().sum()
        if float(mt_sq.detach().cpu()) <= self.eps:
            return torch.zeros_like(grad_anchor)
        dot = (grad_mt * grad_anchor).sum()
        projected = grad_anchor
        if float(dot.detach().cpu()) < 0:
            projected = grad_anchor - (dot / (mt_sq + self.eps)) * grad_mt
        mt_norm = mt_sq.sqrt()
        projected_norm = projected.square().sum().sqrt()
        scale = torch.clamp(
            self.grad_cap * mt_norm / (projected_norm + self.eps), max=1.0
        )
        return projected * scale * expanded_mask

    def route_gradients(
        self,
        grad_mt: torch.Tensor,
        grad_anchor: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Return the auxiliary gradient allowed to reach the encoder."""
        grad_mt_f = grad_mt.detach().float()
        grad_anchor_f = grad_anchor.detach().float()
        valid_mask = source_mask.bool()
        expanded_mask = valid_mask.unsqueeze(-1).to(grad_mt_f.dtype)
        grad_mt_f = grad_mt_f * expanded_mask
        grad_anchor_f = grad_anchor_f * expanded_mask

        if self.gradient_policy == "token_project":
            safe_anchor = self._token_project(grad_mt_f, grad_anchor_f, valid_mask)
        elif self.gradient_policy == "global_project":
            safe_anchor = self._global_project(grad_mt_f, grad_anchor_f, valid_mask)
        elif self.gradient_policy == "joint":
            safe_anchor = grad_anchor_f
        else:
            safe_anchor = torch.zeros_like(grad_anchor_f)

        mt_norm = grad_mt_f.square().sum(dim=-1).sqrt()
        anchor_norm = grad_anchor_f.square().sum(dim=-1).sqrt()
        safe_norm = safe_anchor.square().sum(dim=-1).sqrt()
        dot_before = (grad_mt_f * grad_anchor_f).sum(dim=-1)
        dot_after = (grad_mt_f * safe_anchor).sum(dim=-1)
        denom_before = mt_norm * anchor_norm + self.eps
        denom_after = mt_norm * safe_norm + self.eps
        cosine_before = dot_before / denom_before
        cosine_after = dot_after / denom_after
        valid_grad = valid_mask & (mt_norm > self.eps)
        conflict = valid_grad & (dot_before < 0)

        count = int(valid_grad.sum().detach().cpu())
        if count:
            ratio = safe_norm[valid_grad] / (mt_norm[valid_grad] + self.eps)
            stats = {
                "valid_tokens": count,
                "conflict_tokens": int(conflict.sum().detach().cpu()),
                "conflict_rate": float(conflict.float().sum().detach().cpu()) / count,
                "cosine_before": float(cosine_before[valid_grad].mean().detach().cpu()),
                "cosine_after": float(cosine_after[valid_grad].mean().detach().cpu()),
                "safe_to_mt_ratio": float(ratio.mean().detach().cpu()),
            }
        else:
            stats = {
                "valid_tokens": 0,
                "conflict_tokens": 0,
                "conflict_rate": 0.0,
                "cosine_before": 0.0,
                "cosine_after": 0.0,
                "safe_to_mt_ratio": 0.0,
            }
        return safe_anchor.to(dtype=grad_mt.dtype), stats

    def _ramp(self, global_step: int) -> float:
        if self.warmup_ratio == 0:
            return 1.0
        return min(1.0, max(0.0, float(global_step) / self.warmup_steps))

    def forward(
        self,
        loss_mt: torch.Tensor,
        model_outputs: Dict[str, torch.Tensor],
        global_step: int,
    ) -> Dict[str, object]:
        if not torch.is_grad_enabled() or model_outputs.get("projected_source") is None:
            return {
                "loss": loss_mt,
                "log_dict": {"loss_mt": float(loss_mt.detach().cpu())},
            }

        source_hidden = model_outputs["source_hidden"]
        encoder_proxy = model_outputs["encoder_proxy"]
        anchor_loss, anchor_stats = self.compute_anchor_loss(
            model_outputs["projected_source"],
            model_outputs["teacher_hidden"],
            model_outputs["source_mask"],
            model_outputs["target_mask"],
        )

        grad_mt = torch.autograd.grad(
            loss_mt, encoder_proxy, retain_graph=True, create_graph=False
        )[0]
        grad_anchor = torch.autograd.grad(
            anchor_loss, encoder_proxy, retain_graph=True, create_graph=False
        )[0]
        safe_anchor, route_stats = self.route_gradients(
            grad_mt, grad_anchor, model_outputs["source_mask"]
        )
        ramp = self._ramp(global_step)
        encoder_gradient = grad_mt.detach() + ramp * safe_anchor.detach()

        # Numerically zero, but its derivative reconnects the routed gradient to
        # the real encoder graph. Decoder/projector gradients still come from
        # the two ordinary losses below.
        surrogate = (source_hidden.float() * encoder_gradient.float()).sum()
        zero_value_surrogate = surrogate - surrogate.detach()
        total_loss = loss_mt + anchor_loss + zero_value_surrogate

        self._diagnostics["batches"] += 1
        self._diagnostics["valid_tokens"] += route_stats["valid_tokens"]
        self._diagnostics["conflict_tokens"] += route_stats["conflict_tokens"]
        count = route_stats["valid_tokens"]
        self._diagnostics["cosine_before_sum"] += route_stats["cosine_before"] * count
        self._diagnostics["cosine_after_sum"] += route_stats["cosine_after"] * count
        self._diagnostics["safe_to_mt_ratio_sum"] += route_stats["safe_to_mt_ratio"] * count
        self._diagnostics["anchor_loss_sum"] += float(anchor_loss.detach().cpu())
        self._diagnostics["alignment_entropy_sum"] += anchor_stats["mean_alignment_entropy"]
        self._diagnostics["anchor_weight_sum"] += anchor_stats["mean_anchor_weight"]
        self._diagnostics["source_length_sum"] += float(
            model_outputs["source_mask"].sum(dim=-1).float().mean().detach().cpu()
        )
        self._diagnostics["ramp_sum"] += ramp

        log_dict = {
            "loss_mt": round(float(loss_mt.detach().cpu()), 6),
            "loss_anchor": round(float(anchor_loss.detach().cpu()), 6),
            "loss_total": round(float((loss_mt + anchor_loss).detach().cpu()), 6),
            "ramp": round(ramp, 6),
            **{key: round(value, 6) for key, value in route_stats.items()},
            **{key: round(value, 6) for key, value in anchor_stats.items()},
        }
        return {"loss": total_loss, "log_dict": log_dict}

    def get_diagnostics(self) -> Dict[str, object]:
        batches = int(self._diagnostics["batches"])
        tokens = int(self._diagnostics["valid_tokens"])
        return {
            "gradient_policy": self.gradient_policy,
            "grad_cap": self.grad_cap,
            "warmup_ratio": self.warmup_ratio,
            "total_steps": self.total_steps,
            "observed_batches": batches,
            "observed_valid_tokens": tokens,
            "conflict_tokens": int(self._diagnostics["conflict_tokens"]),
            "conflict_rate": self._diagnostics["conflict_tokens"] / max(1, tokens),
            "mean_cosine_before": self._diagnostics["cosine_before_sum"] / max(1, tokens),
            "mean_cosine_after": self._diagnostics["cosine_after_sum"] / max(1, tokens),
            "mean_safe_to_mt_ratio": self._diagnostics["safe_to_mt_ratio_sum"] / max(1, tokens),
            "mean_anchor_loss": self._diagnostics["anchor_loss_sum"] / max(1, batches),
            "mean_alignment_entropy": self._diagnostics["alignment_entropy_sum"] / max(1, batches),
            "mean_anchor_weight": self._diagnostics["anchor_weight_sum"] / max(1, batches),
            "mean_source_length": self._diagnostics["source_length_sum"] / max(1, batches),
            "mean_ramp": self._diagnostics["ramp_sum"] / max(1, batches),
        }

    def save_diagnostics(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        with open(
            os.path.join(output_dir, "gradient_diagnostics.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(self.get_diagnostics(), handle, indent=2, ensure_ascii=False)
