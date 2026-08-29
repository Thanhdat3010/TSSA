"""
TSSA Seq2Seq Model Architecture
Integrates Pretrained BARTpho Seq2Seq backbone with Head-Wise Router
and Teacher representations.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM
from .head_router import HeadWiseRouter

class TSSASeq2SeqModel(nn.Module):
    def __init__(self, model_name_or_path: str = "vinai/bartpho-syllable", use_route: bool = True,
                 d_model: int = None, n_heads: int = None, n_decoder_layers: int = None):
        super().__init__()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, use_safetensors=True)
        self.use_route = use_route
        
        # Dynamically read architectural dimensions from backbone model config
        cfg = self.model.config
        self.d_model = d_model or getattr(cfg, "d_model", getattr(cfg, "hidden_size", 1024))
        self.n_heads = n_heads or getattr(cfg, "decoder_attention_heads", getattr(cfg, "num_attention_heads", 16))
        self.n_decoder_layers = n_decoder_layers or getattr(cfg, "decoder_layers", getattr(cfg, "num_decoder_layers", 12))

        if use_route:
            self.router = HeadWiseRouter(d_model=self.d_model, n_heads=self.n_heads, n_layers=self.n_decoder_layers)
        else:
            self.router = None

        self.last_router_gates = None

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                labels: torch.Tensor = None, decoder_attention_mask: torch.Tensor = None,
                output_attentions: bool = True, output_hidden_states: bool = True, **kwargs):
        """
        Forward pass for student model.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True
        )

        # If Router is active, compute router gate activations on decoder states
        if self.use_route and self.router is not None and outputs.decoder_hidden_states is not None:
            dec_last_state = outputs.decoder_hidden_states[-1] # [B, T, D]
            # Context approximation from encoder state mean
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
            "router_gates": self.last_router_gates
        }

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
    def warnings_issued(self):
        return getattr(self.model, "warnings_issued", {})

    def can_generate(self):
        return True

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.model.prepare_inputs_for_generation(*args, **kwargs)

    def save_pretrained(self, save_directory, **kwargs):
        self.model.save_pretrained(save_directory, **kwargs)
        if self.router is not None:
            import os
            torch.save(self.router.state_dict(), os.path.join(save_directory, "router.pt"))

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        """Standard Seq2Seq generation pass."""
        return self.model.generate(*args, **kwargs)

    def prune_heads(self, pruned_indices: list):
        if self.router is not None:
            self.router.prune_heads(pruned_indices)

    def reset_pruning_mask(self):
        if self.router is not None:
            self.router.reset_pruning_mask()
