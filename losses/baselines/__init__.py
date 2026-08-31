"""
Baseline Alignment Losses Module
Exports all 8 modular alignment baselines organized into dedicated folders:
1. align_to_distill
2. structural_supervision
3. shift_aet
4. cross_init
5. awesome_align
6. dm_bli
7. cl_lsa
8. dpo_align
"""

from .align_to_distill import AlignToDistillLoss
from .structural_supervision import StructuralSupervisionLoss
from .shift_aet import ShiftAETLoss
from .cross_init import CrossInitLoss
from .awesome_align import AwesomeAlignLoss
from .dm_bli import DMBLISubspaceLoss
from .cl_lsa import CrossLingualInfoNCELoss
from .dpo_align import AlignmentDPOLoss
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
    "UnifiedAlignmentLossFactory"
]
