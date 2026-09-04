"""
TSSA ViT5 Model Architecture (Dedicated Decoupled Implementation)
Integrates Pretrained ViT5 (T5ForConditionalGeneration) backbone with:
1. Target-Guided Cross-Attention Anchoring (extracted directly from T5 EncDecAttention blocks)
2. Residual Cross-Lingual Semantic Projector (adapted for T5 hidden dimensions)
3. Dynamic Head-Wise Attention Router (adapted for T5 heads/layers)
4. Online Frozen Teacher Semantic Anchoring (via T5 Encoder)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM
from .head_router import HeadWiseRouter
from .semantic_projector import ResidualSemanticProjector

class TSSAViT5Model(nn.Module):
    # Hugging Face Trainer serialization attributes
    _keys_to_ignore_on_save = None
    _keys_to_ignore_on_load_missing = None
    _keys_to_ignore_on_load_unexpected = None

    def __init__(self, model_name_or_path: str = "VietAI/vit5-base", use_route: bool = True,
                 d_model: int = None, n_heads: int = None, n_decoder_layers: int = None):
        super().__init__()
        # Load backbone T5 model
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        self.use_route = use_route

        # Dynamically read architectural dimensions from T5 config
        cfg = self.model.config
        self.d_model = d_model or getattr(cfg, "d_model", 768)
        self.n_heads = n_heads or getattr(cfg, "num_heads", 12)
        self.n_decoder_layers = n_decoder_layers or getattr(cfg, "num_decoder_layers", 12)
        self.d_kv = getattr(cfg, "d_kv", self.d_model // self.n_heads)

        # 1. Residual Cross-Lingual Semantic Projector
        self.projector = ResidualSemanticProjector(d_model=self.d_model, hidden_dim=self.d_model * 2)

        # 2. Dynamic Head-Wise Attention Router
        if use_route:
            self.router = HeadWiseRouter(d_model=self.d_model, n_heads=self.n_heads, n_layers=self.n_decoder_layers)
        else:
            self.router = None

        self.last_router_gates = None

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                labels: torch.Tensor = None, decoder_attention_mask: torch.Tensor = None,
                output_attentions: bool = True, output_hidden_states: bool = True, **kwargs):
        """
        Forward pass for student model with TSSA 2.0 components adapted for T5.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        teacher_enc_states = None
        teacher_sent_vec = None
        align_matrix_ts = None
        student_projected_sent = None
        cross_attentions_tuple = None

        # 1. Source Sentence Pooling & Residual Projection
        if outputs.encoder_last_hidden_state is not None:
            mask_src = attention_mask.unsqueeze(-1).float()
            student_sent_raw = (outputs.encoder_last_hidden_state * mask_src).sum(dim=1) / mask_src.sum(dim=1).clamp(min=1.0)
            student_projected_sent = self.projector(student_sent_raw) # [B, D]

        # 2. Online Frozen Teacher Extraction & Target-to-Source Semantic Alignment Matrix
        if labels is not None and self.training:
            with torch.no_grad():
                pad_id = getattr(self.model.config, "pad_token_id", 0)
                tgt_ids = labels.clone()
                tgt_ids[tgt_ids == -100] = pad_id
                tgt_mask = (tgt_ids != pad_id).long()

                # T5 encoder call
                teacher_out = self.model.encoder(
                    input_ids=tgt_ids,
                    attention_mask=tgt_mask,
                    return_dict=True
                )
                teacher_enc_states = teacher_out.last_hidden_state.detach() # [B, T, D]

                # Masked mean sentence pooling for Teacher
                mask_tgt = tgt_mask.unsqueeze(-1).float()
                teacher_sent_vec = (teacher_enc_states * mask_tgt).sum(dim=1) / mask_tgt.sum(dim=1).clamp(min=1.0) # [B, D]

                # Target-to-Source Subword Semantic Alignment Posterior Matrix A [B, T, S]
                tgt_norm = F.normalize(teacher_enc_states, p=2, dim=-1) # [B, T, D]
                src_norm = F.normalize(outputs.encoder_last_hidden_state.detach(), p=2, dim=-1) # [B, S, D]
                sim_ts = torch.bmm(tgt_norm, src_norm.transpose(1, 2)) / 0.5 # [B, T, S]
                align_matrix_ts = F.softmax(sim_ts, dim=-1).detach() # [B, T, S]

        # 3. Direct Exact Cross-Attention Computation for Top T5 Decoder Blocks
        if self.training and outputs.decoder_hidden_states is not None and outputs.encoder_last_hidden_state is not None:
            B, T, D = outputs.decoder_hidden_states[-1].shape
            S = outputs.encoder_last_hidden_state.size(1)
            H = self.n_heads
            d_k = self.d_kv
            enc_state = outputs.encoder_last_hidden_state # [B, S, D]

            num_layers = min(self.n_decoder_layers, len(outputs.decoder_hidden_states) - 1)
            top_3_start = max(0, num_layers - 3)
            cross_attn_list = []
            for l in range(top_3_start, num_layers):
                dec_state = outputs.decoder_hidden_states[l] # [B, T, D]
                
                # In T5: block[l].layer[1] is EncDecAttention
                t5_block = self.model.decoder.block[l]
                enc_dec_attn = t5_block.layer[1].EncDecAttention

                q = enc_dec_attn.q(dec_state).view(B, T, H, d_k).transpose(1, 2) # [B, H, T, d_k]
                k = enc_dec_attn.k(enc_state).view(B, S, H, d_k).transpose(1, 2) # [B, H, S, d_k]
                scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5) # [B, H, T, S]
                attn_map = F.softmax(scores, dim=-1) # [B, H, T, S]
                cross_attn_list.append(attn_map)
            cross_attentions_tuple = tuple(cross_attn_list)

        # 4. Dynamic Head-Wise Router Gate Activations
        if self.use_route and self.router is not None and outputs.decoder_hidden_states is not None:
            dec_last_state = outputs.decoder_hidden_states[-1] # [B, T, D]
            enc_last_state = outputs.encoder_last_hidden_state # [B, S, D]
            ctx_approx = enc_last_state.mean(dim=1, keepdim=True).expand(-1, dec_last_state.size(1), -1) # [B, T, D]

            gates_list = []
            for l in range(min(self.n_decoder_layers, len(outputs.decoder_hidden_states) - 1)):
                layer_gate = self.router(dec_last_state, ctx_approx, layer_idx=l) # [B, H, T, 1]
                gates_list.append(layer_gate)
                
            self.last_router_gates = torch.stack(gates_list, dim=1) # [B, L, H, T, 1]
        else:
            self.last_router_gates = None

        # Pack custom outputs (identical interface for TSSAUnifiedCriterion)
        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
            "encoder_last_hidden_state": outputs.encoder_last_hidden_state,
            "encoder_hidden_states": outputs.encoder_hidden_states,
            "decoder_hidden_states": outputs.decoder_hidden_states,
            "cross_attentions": cross_attentions_tuple,
            "router_gates": self.last_router_gates,
            "student_projected_sent": student_projected_sent,
            "teacher_enc_states": teacher_enc_states,
            "teacher_sent_vec": teacher_sent_vec,
            "align_matrix_ts": align_matrix_ts
        }

    def generate(self, *args, **kwargs):
        """Delegates sequence generation directly to the inner Seq2Seq model."""
        return self.model.generate(*args, **kwargs)

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, *args, **kwargs):
        return self.model.resize_token_embeddings(*args, **kwargs)

    def save_pretrained(self, save_directory, **kwargs):
        return self.model.save_pretrained(save_directory, **kwargs)

    @property
    def config(self):
        return self.model.config

    @property
    def generation_config(self):
        return self.model.generation_config

    @generation_config.setter
    def generation_config(self, val):
        self.model.generation_config = val

    @property
    def device(self):
        return self.model.device

    @property
    def main_input_name(self):
        return getattr(self.model, "main_input_name", "input_ids")

    @property
    def warnings_issued(self):
        return getattr(self.model, "warnings_issued", {})

    def can_generate(self):
        return True

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.model.prepare_inputs_for_generation(*args, **kwargs)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if hasattr(self, "model") and hasattr(self.model, name):
                return getattr(self.model, name)
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
