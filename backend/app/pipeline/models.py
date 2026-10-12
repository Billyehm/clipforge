"""Shared data contracts for the clipforge pipeline.

Every stage reads and writes these models, so treat them as stable.
Other bits of the project depend on the exact field names here.
"""

from pydantic import BaseModel, Field


class Word(BaseModel):
    text: str
    start: float
    end: float


class Segment(BaseModel):
    id: int
    start: float
    end: float
    text: str
    speaker: str | None = None
    words: list[Word] = []


class Transcript(BaseModel):
    video_id: str
    duration: float
    language: str = "en"
    segments: list[Segment] = []


class Candidate(BaseModel):
    id: int
    start: float
    end: float
    score: float = Field(ge=0, le=100)
    hook_line: str
    reason: str
    title: str
    hashtags: list[str] = []
    speaker: str | None = None


class ClipRecord(BaseModel):
    id: str
    video_id: str
    candidate: Candidate
    status: str = "pending"
    output_path: str | None = None
    caption_path: str | None = None
    error: str | None = None
