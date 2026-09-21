"""
models/tssa_v4_seq2seq.py
TSSA-V4 Architecture: Decoupled Middle-Layer Semantic Anchoring
Key Innovations:
1. Dynamic Middle-Layer Tap: mid_layer = config.encoder_layers // 2 (Layer 3 for 6-layer BARTpho).
   - Layers 1 to mid_layer learn surface forms & abstract cross-lingual semantics.
   - Layers (mid_layer+1) to final remain 100% free for MT syntax & Cross-Attention prep.
2. Decoupled 2-Layer MLP Projection Head:
   - Alignment loss is computed exclusively in the projected latent space z = MLP(H_mid).
   - Gradients do not directly deform the backbone's representation manifold.
3. Exposes both Middle-Layer states and Final-Layer states for hybrid objectives.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM

class DecoupledProjector(nn.Module):
    """2-Layer Non-Linear MLP Projector to decouple alignment from translation."""
    def __init__(self, d_model: int, hidden_dim: int = None, out_dim: int = None):
        super().__init__()
        hidden_dim = hidden_dim or d_model * 2
        out_dim = out_dim or d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_norm(self.net(x))

class TSSAV4Seq2SeqModel(nn.Module):
    _keys_to_ignore_on_save = None
    _keys_to_ignore_on_load_missing = None
    _keys_to_ignore_on_load_unexpected = None

    def __init__(self, model_name_or_path: str = "vinai/bartpho-syllable", **kwargs):
        super().__init__()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, use_safetensors=True)
        cfg = self.model.config

        self.d_model = getattr(cfg, "d_model", getattr(cfg, "hidden_size", 1024))
        self.num_enc_layers = getattr(cfg, "encoder_layers", getattr(cfg, "num_encoder_layers", 6))
        
        # Dynamic Middle Layer: mid_layer = num_enc_layers // 2
        # For 6-layer BARTpho: mid_layer = 3
        # For 12-layer models: mid_layer = 6
        self.mid_layer = max(1, self.num_enc_layers // 2)

        # Decoupled Projector for Middle Layer States
        self.mid_projector = DecoupledProjector(d_model=self.d_model, hidden_dim=self.d_model * 2, out_dim=self.d_model)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                labels: torch.Tensor = None, decoder_attention_mask: torch.Tensor = None,
                output_attentions: bool = True, output_hidden_states: bool = True, **kwargs):
        """
        Forward pass extracting both middle-layer projected representations
        and final MT generation states.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
            output_hidden_states=True,
            output_attentions=output_attentions,
            return_dict=True
        )

        student_mid_h = None
        student_mid_z = None
        teacher_mid_h = None
        teacher_mid_z = None
        teacher_enc_states = None

        # 1. Trích xuất Student Middle Layer State (Index mid_layer trong tuple hidden_states)
        if outputs.encoder_hidden_states is not None and len(outputs.encoder_hidden_states) > self.mid_layer:
            student_mid_h = outputs.encoder_hidden_states[self.mid_layer] # [B, S, D]
            student_mid_z = self.mid_projector(student_mid_h)            # [B, S, D]

        # 2. Trích xuất Teacher Middle Layer State (Target Vietnamese qua Frozen Encoder)
        if labels is not None and self.training:
            with torch.no_grad():
                pad_id = getattr(self.model.config, "pad_token_id", 1)
                tgt_ids = labels.clone()
                tgt_ids[tgt_ids == -100] = pad_id
                tgt_mask = (tgt_ids != pad_id).long()

                # Gọi trực tiếp backbone encoder với output_hidden_states=True
                encoder_module = self.model.get_encoder() if hasattr(self.model, "get_encoder") else self.model.model.encoder
                teacher_out = encoder_module(
                    input_ids=tgt_ids,
                    attention_mask=tgt_mask,
                    output_hidden_states=True,
                    return_dict=True
                )
                teacher_enc_states = teacher_out.last_hidden_state.detach()

                if teacher_out.hidden_states is not None and len(teacher_out.hidden_states) > self.mid_layer:
                    teacher_mid_h = teacher_out.hidden_states[self.mid_layer].detach() # [B, T, D]
                    # Chiếu qua projector (stop-gradient để Teacher hoàn toàn cố định)
                    teacher_mid_z = self.mid_projector(teacher_mid_h).detach()         # [B, T, D]

        # Đóng gói toàn bộ thông tin cần thiết vào output dictionary
        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
            "encoder_last_hidden_state": outputs.encoder_last_hidden_state,
            "encoder_hidden_states": outputs.encoder_hidden_states,
            "decoder_hidden_states": outputs.decoder_hidden_states,
            "cross_attentions": outputs.cross_attentions,
            "student_mid_h": student_mid_h,
            "student_mid_z": student_mid_z,
            "teacher_mid_h": teacher_mid_h,
            "teacher_mid_z": teacher_mid_z,
            "teacher_enc_states": teacher_enc_states
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

