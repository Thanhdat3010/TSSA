"""
losses/v4_criterion.py
TSSA-V4 Alignment Criterion:
1. Sentence-Level InfoNCE with Memory Queue (MoCo-style 256 negatives):
   - Solves the small-batch (B=16 -> 15 negatives) bottleneck pointed out by Senior Review.
   - Maintains a FIFO queue of 256 target sentence vectors from recent batches.
   - True cross-lingual semantic contrastive alignment with tau=0.07.
2. Token-Level Middle-Layer Barycenter Anchoring:
   - Dynamic entropy gating w_s = exp(-H_norm / tau_H).
   - Operates on projected latent space z = MLP(H_mid).
3. Modes:
   - v4_sent   : Only Sentence-Level InfoNCE + Memory Queue (lambda_sent = 0.10)
   - v4_tok    : Only Token-Level Middle Barycenter (lambda_tok = 0.10)
   - v4_hybrid : Dual-Level Fusion (lambda_sent = 0.08, lambda_tok = 0.05)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class V4AlignmentCriterion(nn.Module):
    def __init__(self,
                 mode: str = "v4_sent",
                 d_model: int = 1024,
                 queue_size: int = 256,
                 temperature: float = 0.07,
                 align_tau: float = 0.10,
                 entropy_tau: float = 0.50,
                 lambda_sent: float = 0.10,
                 lambda_tok: float = 0.10,
                 eps: float = 1e-8):
        super().__init__()
        self.mode = mode.lower().strip()
        self.d_model = d_model
        self.queue_size = queue_size
        self.temperature = temperature
        self.align_tau = align_tau
        self.entropy_tau = entropy_tau
        self.lambda_sent = lambda_sent
        self.lambda_tok = lambda_tok
        self.eps = eps

        # Memory Queue of target sentence vectors for contrastive learning (MoCo style)
        # Guarantees >= 256 stable negatives at EVERY step even with batch_size=16!
        self.register_buffer("queue", F.normalize(torch.randn(queue_size, d_model), p=2, dim=-1))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        """Updates FIFO memory queue with newly computed target sentence vectors."""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr[0])

        if ptr + batch_size <= self.queue_size:
            self.queue[ptr:ptr + batch_size] = keys
            ptr = (ptr + batch_size) % self.queue_size
        else:
            rem = self.queue_size - ptr
            self.queue[ptr:] = keys[:rem]
            self.queue[:batch_size - rem] = keys[rem:]
            ptr = batch_size - rem

        self.queue_ptr[0] = ptr

    def compute_sentence_infonce(self, student_z: torch.Tensor, teacher_z: torch.Tensor,
                                 src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        """
        Computes Sentence-Level InfoNCE loss with in-batch positives + in-batch negatives + queue negatives.
        """
        # 1. Masked Mean-Pooling for Sentence Vectors
        if src_mask is not None:
            mask_s = src_mask.unsqueeze(-1).float()
            u_src = (student_z * mask_s).sum(dim=1) / mask_s.sum(dim=1).clamp(min=1.0)
        else:
            u_src = student_z.mean(dim=1)

        if tgt_mask is not None:
            mask_t = tgt_mask.unsqueeze(-1).float()
            v_tgt = (teacher_z * mask_t).sum(dim=1) / mask_t.sum(dim=1).clamp(min=1.0)
        else:
            v_tgt = teacher_z.mean(dim=1)

        # 2. L2 Normalization onto Unit Hypersphere
        u_norm = F.normalize(u_src.float(), p=2, dim=-1) # [B, D]
        v_norm = F.normalize(v_tgt.float(), p=2, dim=-1) # [B, D]

        B, D = u_norm.shape

        # 3. Positive Logits: u_i . v_i [B, 1]
        l_pos = (u_norm * v_norm).sum(dim=-1, keepdim=True) / self.temperature # [B, 1]

        # 4. In-batch Negatives: u_i . v_j for j != i [B, B]
        l_in_batch = torch.matmul(u_norm, v_norm.transpose(0, 1)) / self.temperature # [B, B]

        # 5. Memory Queue Negatives: u_i . q_k [B, Q] (256 negatives)
        queue_detached = self.queue.clone().detach().to(u_norm.device)
        l_queue = torch.matmul(u_norm, queue_detached.transpose(0, 1)) / self.temperature # [B, Q]

        # Concatenate: [B, 1 + B + Q]
        # Labels for CrossEntropy: index of positive pair (diagonal in l_in_batch)
        logits = torch.cat([l_in_batch, l_queue], dim=1) # [B, B + Q]
        labels = torch.arange(B, device=logits.device, dtype=torch.long)

        loss_nce = F.cross_entropy(logits, labels)

        # Enqueue current teacher vectors into memory bank for future steps
        if self.training:
            self._dequeue_and_enqueue(v_norm.detach())

        return loss_nce

    def compute_token_barycenter(self, student_z: torch.Tensor, teacher_z: torch.Tensor,
                                 src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        """Computes Token-Level Middle Soft Barycenter loss on projected embeddings."""
        s_norm = F.normalize(student_z.float(), p=2, dim=-1) # [B, S, D]
        t_norm = F.normalize(teacher_z.float(), p=2, dim=-1) # [B, T, D]

        B, S, D = s_norm.shape
        T = t_norm.size(1)

        # Similarity Matrix [B, S, T]
        sim_st = torch.bmm(s_norm, t_norm.transpose(1, 2)) / self.align_tau

        if tgt_mask is not None:
            mask_t = (1.0 - tgt_mask.unsqueeze(1).float()) * -1e4
            sim_st = sim_st + mask_t

        # Soft Alignment Posterior with stop-gradient
        align_st = F.softmax(sim_st, dim=-1).detach() # [B, S, T]

        # Spherical Target Barycenter Vector
        target_barycenter = F.normalize(torch.bmm(align_st, t_norm), p=2, dim=-1, eps=self.eps) # [B, S, D]

        # Dynamic Information Entropy Gate
        p_clamped = align_st.clamp(min=self.eps)
        entropy_s = - (p_clamped * torch.log(p_clamped)).sum(dim=-1) # [B, S]
        
        tgt_len = tgt_mask.sum(dim=-1, keepdim=True).float().clamp(min=2.0) if tgt_mask is not None else torch.full((B, 1), float(T), device=s_norm.device)
        max_entropy = torch.log(tgt_len)
        norm_entropy_s = (entropy_s / max_entropy).clamp(0.0, 1.0)
        w_s = torch.exp(- norm_entropy_s / self.entropy_tau) # [B, S]

        valid_mask = src_mask.float() if src_mask is not None else torch.ones((B, S), device=s_norm.device)

        # Cosine Distance: 1 - <s_norm, target_barycenter>
        cos_dist = 1.0 - (s_norm * target_barycenter).sum(dim=-1) # [B, S] in [0, 2]
        weighted_loss = cos_dist * w_s * valid_mask

        normalizer = valid_mask.sum().clamp(min=1.0)
        loss_tok = weighted_loss.sum() / normalizer
        return loss_tok

    def forward(self, loss_mt: torch.Tensor, student_outputs: dict, batch: dict, **kwargs) -> dict:
        """
        Computes V4 multi-task loss combining MT translation with middle-layer alignment.
        """
        student_mid_z = student_outputs.get("student_mid_z")
        teacher_mid_z = student_outputs.get("teacher_mid_z")

        src_mask = batch.get("attention_mask")
        tgt_mask = batch.get("decoder_attention_mask")

        loss_sent = torch.tensor(0.0, device=loss_mt.device)
        loss_tok = torch.tensor(0.0, device=loss_mt.device)
        total_loss = loss_mt

        # Nếu có đủ representations của student và teacher
        if student_mid_z is not None and teacher_mid_z is not None:
            if self.mode == "v4_sent":
                loss_sent = self.compute_sentence_infonce(student_mid_z, teacher_mid_z, src_mask, tgt_mask)
                total_loss = loss_mt + (self.lambda_sent * loss_sent)

            elif self.mode == "v4_tok":
                loss_tok = self.compute_token_barycenter(student_mid_z, teacher_mid_z, src_mask, tgt_mask)
                total_loss = loss_mt + (self.lambda_tok * loss_tok)

            elif self.mode == "v4_hybrid":
                loss_sent = self.compute_sentence_infonce(student_mid_z, teacher_mid_z, src_mask, tgt_mask)
                loss_tok = self.compute_token_barycenter(student_mid_z, teacher_mid_z, src_mask, tgt_mask)
                total_loss = loss_mt + (0.08 * loss_sent) + (0.05 * loss_tok)

        log_dict = {
            "loss_mt": round(loss_mt.item(), 4),
            "loss_sent": round(loss_sent.item(), 4) if isinstance(loss_sent, torch.Tensor) else 0.0,
            "loss_tok": round(loss_tok.item(), 4) if isinstance(loss_tok, torch.Tensor) else 0.0,
            "loss_total": round(total_loss.item(), 4)
        }

        return {"loss": total_loss, "log_dict": log_dict}
