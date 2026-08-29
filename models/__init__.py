"""Models module for TSSA and Teacher architectures."""
from .teacher_wrapper import TeacherWrapper
from .head_router import HeadWiseRouter
from .tssa_seq2seq import TSSASeq2SeqModel

__all__ = [
    "TeacherWrapper",
    "HeadWiseRouter",
    "TSSASeq2SeqModel",
]
