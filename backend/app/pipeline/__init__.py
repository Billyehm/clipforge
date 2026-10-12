"""Public interface of the pipeline package."""

from .models import Candidate, ClipRecord, Segment, Transcript, Word
from .cache import StageCache
from .score import score_transcript
from .select import dedupe, pick_top
from .captions import words_to_ass
from .reframe import plan_reframe

__all__ = [
    "Candidate",
    "ClipRecord",
    "Segment",
    "StageCache",
    "Transcript",
    "Word",
    "dedupe",
    "pick_top",
    "plan_reframe",
    "score_transcript",
    "words_to_ass",
]
