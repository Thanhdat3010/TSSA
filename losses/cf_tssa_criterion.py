"""Finite-intervention, semantic-specific anchor selection for CF-TSSA."""

from __future__ import annotations

import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class CFTSSACriterion(nn.Module):
    def __init__(self, total_steps: int, tau: float = 0.10, eta: float = 0.10,
                 margin: float = 1e-4, grad_budget: float = 0.05,
                 warmup_ratio: float = 0.10):
        super().__init__()
        self.total_steps = max(1, int(total_steps))
        self.tau, self.eta, self.margin = tau, eta, margin
        self.grad_budget, self.warmup_ratio = grad_budget, warmup_ratio
        self.stats = dict(observed_batches=0, candidates=0, accepted=0, accepted_batches=0,
                          sum_true_uplift=0.0, sum_control_uplift=0.0,
                          sum_cosine_gap=0.0, sum_grad_ratio=0.0)

    @staticmethod
    def _unit(vector: torch.Tensor) -> torch.Tensor:
        return F.normalize(vector.float(), dim=-1, eps=1e-8)

    def build_candidates(self, outputs: dict):
        """At most one source token and one cross-sentence negative per sentence."""
        hidden = outputs["source_hidden"]
        teacher_source = outputs["teacher_source"].float()
        teacher_target = outputs["teacher_target"].float()
        source_mask = outputs["source_mask"]
        target_mask = outputs["target_mask"]
        target_ids = outputs["target_ids"]
        candidates = []
        if hidden.size(0) < 2:
            return candidates
        for row in range(hidden.size(0)):
            src_idx = source_mask[row].nonzero(as_tuple=True)[0]
            tgt_idx = target_mask[row].nonzero(as_tuple=True)[0]
            if not len(src_idx) or not len(tgt_idx):
                continue
            f = teacher_source[row, src_idx]
            t = teacher_target[row, tgt_idx]
            sim = self._unit(f) @ self._unit(t).T / self.tau
            mutual = sim.softmax(dim=1) * sim.softmax(dim=0)
            chosen = mutual.sum(dim=1).argmax()
            source_index = int(src_idx[chosen])
            anchor = (mutual[chosen, :, None] * t).sum(dim=0) / mutual[chosen].sum().clamp_min(1e-8)
            if anchor.norm() < 1e-8:
                continue
            target_cosine = F.cosine_similarity(f[chosen], anchor, dim=0)
            negatives = []
            for other in range(hidden.size(0)):
                if other == row or torch.equal(target_ids[other], target_ids[row]):
                    continue
                other_idx = target_mask[other].nonzero(as_tuple=True)[0]
                if len(other_idx):
                    negatives.append(teacher_target[other, other_idx])
            if not negatives:
                continue
            negatives = torch.cat(negatives, dim=0)
            cosines = self._unit(negatives) @ self._unit(f[chosen])
            gaps = (cosines - target_cosine).abs()
            tied = (gaps - gaps.min()).abs().le(1e-7).nonzero(as_tuple=True)[0]
            negative_index = tied[torch.randint(len(tied), (1,), device=tied.device).item()]
            negative = negatives[negative_index]
            h = hidden[row, source_index].detach().float()
            h_norm = h.norm()
            if h_norm < 1e-8 or negative.norm() < 1e-8:
                continue
            anchor = anchor * (h_norm / anchor.norm())
            negative = negative * (h_norm / negative.norm())
            true_delta = anchor - h
            wrong_delta = negative - h
            if true_delta.norm() < 1e-8 or wrong_delta.norm() < 1e-8:
                continue
            patch_norm = self.eta * h_norm
            candidates.append(dict(row=row, source_index=source_index, anchor=anchor.detach(),
                                   true_patch=(patch_norm * self._unit(true_delta)).detach(),
                                   wrong_patch=(patch_norm * self._unit(wrong_delta)).detach(),
                                   cosine_gap=float((cosines[negative_index] - target_cosine).abs())))
        return candidates

    @staticmethod
    def sentence_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        losses = F.cross_entropy(logits.float().transpose(1, 2), labels,
                                 reduction="none", ignore_index=-100)
        return losses.sum(dim=1) / (labels != -100).sum(dim=1).clamp_min(1)

    def probe(self, model, outputs: dict, inputs: dict, candidates: list):
        """Three matched teacher-forced decoder passes; no gradient or dropout."""
        hidden = outputs["source_hidden"].detach().float()
        true_hidden, wrong_hidden = hidden.clone(), hidden.clone()
        for item in candidates:
            row, index = item["row"], item["source_index"]
            true_hidden[row, index] += item["true_patch"]
            wrong_hidden[row, index] += item["wrong_patch"]
        backbone = model.backbone.model
        was_training = backbone.training
        backbone.eval()
        try:
            with torch.no_grad(), torch.autocast(device_type=hidden.device.type, enabled=False):
                nlls = []
                for state in (hidden, true_hidden, wrong_hidden):
                    result = model.backbone.decode_from_hidden(
                        state, inputs["attention_mask"], inputs["labels"],
                        inputs.get("decoder_attention_mask"))
                    nlls.append(self.sentence_nll(result.logits, inputs["labels"]))
        finally:
            backbone.train(was_training)
            model.teacher_encoder.eval()
        return nlls[0] - nlls[1], nlls[0] - nlls[2]

    def forward(self, loss_mt, model, model_outputs, inputs, global_step: int):
        self.stats["observed_batches"] += 1
        candidates = self.build_candidates(model_outputs)
        self.stats["candidates"] += len(candidates)
        if not candidates:
            return {"loss": loss_mt, "log_dict": {"cf_accepted": 0, "cf_ratio": 0.0}}
        true_uplift, control_uplift = self.probe(model, model_outputs, inputs, candidates)
        accepted = []
        for item in candidates:
            row = item["row"]
            self.stats["sum_true_uplift"] += float(true_uplift[row])
            self.stats["sum_control_uplift"] += float(control_uplift[row])
            self.stats["sum_cosine_gap"] += item["cosine_gap"]
            if true_uplift[row] > 0 and true_uplift[row] - control_uplift[row] > self.margin:
                accepted.append(item)
        self.stats["accepted"] += len(accepted)
        if not accepted:
            return {"loss": loss_mt, "log_dict": {"cf_accepted": 0, "cf_ratio": 0.0}}
        self.stats["accepted_batches"] += 1
        hidden = model_outputs["source_hidden"]
        terms = []
        for item in accepted:
            h = hidden[item["row"], item["source_index"]].float()
            terms.append((h - item["anchor"]).square().sum() /
                         h.detach().square().sum().clamp_min(1e-8))
        anchor_loss = torch.stack(terms).mean()
        grad_mt = torch.autograd.grad(loss_mt, hidden, retain_graph=True)[0]
        grad_anchor = torch.autograd.grad(anchor_loss, hidden, retain_graph=True)[0]
        mt_norm = grad_mt.detach().float().norm()
        anchor_norm = grad_anchor.detach().float().norm()
        ramp_steps = max(1, round(self.total_steps * self.warmup_ratio))
        ramp = min(1.0, (global_step + 1) / ramp_steps)
        scale = (self.grad_budget * mt_norm / anchor_norm.clamp_min(1e-12)).detach()
        coefficient = scale * ramp
        ratio = float((coefficient * anchor_norm / mt_norm.clamp_min(1e-12)).detach())
        self.stats["sum_grad_ratio"] += ratio
        return {"loss": loss_mt + coefficient * anchor_loss,
                "log_dict": {"cf_accepted": len(accepted), "cf_ratio": ratio,
                             "cf_anchor_loss": float(anchor_loss.detach())}}

    def get_diagnostics(self):
        data = dict(self.stats)
        count = max(1, data["candidates"])
        data.update(accepted_rate=data["accepted"] / count,
                    mean_true_uplift=data.pop("sum_true_uplift") / count,
                    mean_control_uplift=data.pop("sum_control_uplift") / count,
                    mean_cosine_gap=data.pop("sum_cosine_gap") / count,
                    mean_grad_ratio=data.pop("sum_grad_ratio") / max(1, data["accepted_batches"]),
                    tau=self.tau, eta=self.eta, margin=self.margin,
                    grad_budget=self.grad_budget, warmup_ratio=self.warmup_ratio)
        return data

    def save_diagnostics(self, directory):
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "cf_tssa_diagnostics.json"), "w", encoding="utf-8") as stream:
            json.dump(self.get_diagnostics(), stream, indent=2)
