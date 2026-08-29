"""
Optimizer and LR Scheduler Utilities for TSSA
Configures AdamW with weight decay filtering and linear warmup.
"""

import torch
from transformers import get_linear_schedule_with_warmup

def get_optimizer_and_scheduler(model: torch.nn.Module, learning_rate: float = 2e-5,
                                weight_decay: float = 0.01, num_training_steps: int = 10000,
                                warmup_ratio: float = 0.1):
    """
    Constructs AdamW optimizer and linear learning rate scheduler.
    """
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-8)
    num_warmup_steps = int(warmup_ratio * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    return optimizer, scheduler
