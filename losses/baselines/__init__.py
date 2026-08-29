"""Baseline alignment loss functions."""
from .guided_attention_loss import GuidedAttentionLoss
from .joint_align_loss import JointAlignLoss
from .awesome_align_loss import AwesomeAlignLoss
from .cl_lsa_loss import CrossLingualInfoNCELoss

__all__ = [
    "GuidedAttentionLoss",
    "JointAlignLoss",
    "AwesomeAlignLoss",
    "CrossLingualInfoNCELoss",
]
