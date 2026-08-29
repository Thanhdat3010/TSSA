"""
Teacher Wrapper Module for TSSA
Wraps the pretrained Vietnamese Teacher model (BARTpho) and manages gradient flow:
- Mode 'frozen': Stop-gradient sg(.) to serve as an immovable semantic anchor.
- Mode 'trainable': Allows gradients (for Ablation 2 representation drift test).
- Mode 'ema': Updates Teacher with Exponential Moving Average.
"""

import copy
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class TeacherWrapper(nn.Module):
    def __init__(self, model_name: str = "vinai/bartpho-syllable", mode: str = "frozen", ema_decay: float = 0.999):
        super().__init__()
        self.mode = mode
        self.ema_decay = ema_decay
        self.teacher = AutoModel.from_pretrained(model_name, use_safetensors=True)
        
        if mode == "frozen":
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad = False
        elif mode == "ema":
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad = False
            # Keep a shadow copy for EMA updates
            self.shadow_teacher = copy.deepcopy(self.teacher)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict:
        """
        Extracts token states [B, T, D] and sentence pooled vector [B, D].
        """
        if self.mode == "frozen":
            with torch.no_grad():
                outputs = self.teacher.encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                token_states = outputs.last_hidden_state.detach()
        else:
            outputs = self.teacher.encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            token_states = outputs.last_hidden_state

        mask = attention_mask.unsqueeze(-1)
        sent_vec = (token_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        return {
            "token_states": token_states,
            "sent_vec": sent_vec
        }

    @torch.no_grad()
    def update_ema(self, student_model: nn.Module):
        """Updates teacher parameters using exponential moving average of student parameters."""
        if self.mode != "ema":
            return
        for t_param, s_param in zip(self.teacher.parameters(), student_model.parameters()):
            t_param.data.mul_(self.ema_decay).add_(s_param.data, alpha=1.0 - self.ema_decay)
