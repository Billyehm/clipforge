"""Public interface of the pipeline package."""

from .models import Candidate, ClipRecord, Segment, Transcript, Word
from .cache import StageCache
from .score import score_transcript
from .select import dedupe, pick_top
from .captions import words_to_ass, words_to_srt
from .reframe import plan_reframe
from .profiles import GENRE_PROFILES, detect_genre, get_profile

__all__ = [
    "Candidate",
    "ClipRecord",
    "Segment",
    "StageCache",
    "Transcript",
    "Word",
    "dedupe",
    "detect_genre",
    "GENRE_PROFILES",
    "get_profile",
    "pick_top",
    "plan_reframe",
    "score_transcript",
    "words_to_ass",
    "words_to_srt",
]
