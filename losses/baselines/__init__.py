"""
Unified Baseline Alignment Loss Functions (ACL/EMNLP/LREC/EACL Standards)
"""

from .a2d_loss import AlignToDistillLoss
from .structural_supervision_loss import StructuralSupervisionLoss
from .shift_aet_loss import ShiftAETLoss
from .cross_init_loss import CrossInitLoss
from .awesome_align_loss import AwesomeAlignLoss
from .dm_bli_loss import DMBLISubspaceLoss
from .cl_lsa_loss import CrossLingualInfoNCELoss
from .dpo_align_loss import AlignmentDPOLoss
from .factory import UnifiedAlignmentLossFactory

__all__ = [
    "AlignToDistillLoss",
    "StructuralSupervisionLoss",
    "ShiftAETLoss",
    "CrossInitLoss",
    "AwesomeAlignLoss",
    "DMBLISubspaceLoss",
    "CrossLingualInfoNCELoss",
    "AlignmentDPOLoss",
    "UnifiedAlignmentLossFactory",
]
