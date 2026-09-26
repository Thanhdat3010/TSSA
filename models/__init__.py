"""Models module for TSSA and Teacher architectures."""
from .teacher_wrapper import TeacherWrapper
from .head_router import HeadWiseRouter
from .tssa_seq2seq import TSSASeq2SeqModel
from .backbone_adapter import Seq2SeqBackboneAdapter
from .ca_tssa_seq2seq import CATSSASeq2SeqModel

__all__ = [
    "TeacherWrapper",
    "HeadWiseRouter",
    "TSSASeq2SeqModel",
    "Seq2SeqBackboneAdapter",
    "CATSSASeq2SeqModel",
]
