"""
TSSA 2.0 Seq2Seq Model Architecture
Integrates Pretrained BARTpho Seq2Seq backbone with:
1. Target-Guided Cross-Attention Anchoring
2. Residual Cross-Lingual Semantic Projector
3. Dynamic Head-Wise Router
4. Online Frozen Teacher Semantic Anchoring
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM
from .head_router import HeadWiseRouter
from .semantic_projector import ResidualSemanticProjector

class TSSASeq2SeqModel(nn.Module):
    # Hugging Face Trainer serialization attributes
    _keys_to_ignore_on_save = None
    _keys_to_ignore_on_load_missing = None
    _keys_to_ignore_on_load_unexpected = None

    def __init__(self, model_name_or_path: str = "vinai/bartpho-syllable", use_route: bool = True,
                 d_model: int = None, n_heads: int = None, n_decoder_layers: int = None):
        super().__init__()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_path,
            use_safetensors=True,
            attn_implementation="eager"
        )
        self.use_route = use_route
        
        # Dynamically read architectural dimensions from backbone model config
        cfg = self.model.config
        self.d_model = d_model or getattr(cfg, "d_model", getattr(cfg, "hidden_size", 1024))
        self.n_heads = n_heads or getattr(cfg, "decoder_attention_heads", getattr(cfg, "num_attention_heads", 16))
        self.n_decoder_layers = n_decoder_layers or getattr(cfg, "decoder_layers", getattr(cfg, "num_decoder_layers", 12))

        # 1. Residual Cross-Lingual Semantic Projector (TSSA 2.0)
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
        Forward pass for student model with TSSA 2.0 Cross-Attention & Projector components.
        """
        # Always output cross attentions during training for TSSA 2.0
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True
        )

        teacher_enc_states = None
        teacher_sent_vec = None
        align_matrix_ts = None
        student_projected_sent = None

        # 1. Source Sentence Pooling & Residual Projection
        if outputs.encoder_last_hidden_state is not None:
            mask_src = attention_mask.unsqueeze(-1).float()
            student_sent_raw = (outputs.encoder_last_hidden_state * mask_src).sum(dim=1) / mask_src.sum(dim=1).clamp(min=1.0)
            student_projected_sent = self.projector(student_sent_raw) # [B, D]

        # 2. Online Frozen Teacher Extraction & Target-to-Source Semantic Alignment Matrix
        if labels is not None and self.training:
            with torch.no_grad():
                pad_id = getattr(self.model.config, "pad_token_id", 1)
                tgt_ids = labels.clone()
                tgt_ids[tgt_ids == -100] = pad_id
                tgt_mask = (tgt_ids != pad_id).long()

                teacher_out = self.model.model.encoder(
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

        # 3. Dynamic Head-Wise Router Gate Activations
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

        # Pack custom outputs
        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
            "encoder_last_hidden_state": outputs.encoder_last_hidden_state,
            "encoder_hidden_states": outputs.encoder_hidden_states,
            "decoder_hidden_states": outputs.decoder_hidden_states,
            "cross_attentions": outputs.cross_attentions,
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
