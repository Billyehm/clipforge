"""Heuristic viral-moment scorer. No LLM here; that layer comes later.

The scorer slides windows of several lengths over the transcript,
scores each window on five engagement signals, and returns ranked
Candidate objects. It finds promising moments, it does not promise
they will go viral.
"""

import re
from collections import Counter

from .models import Candidate, Segment, Transcript

WINDOW_LENGTHS = (20, 30, 45, 60)
WINDOW_STEP = 5.0
MIN_WINDOW = 15.0

WEIGHTS = {
    "hook": 0.30,
    "emotion": 0.20,
    "density": 0.25,
    "pacing": 0.15,
    "curiosity": 0.10,
}

HOOK_WORDS = ("nobody", "secret", "mistake", "never", "always", "how to", "why")
CONTRARIAN_WORDS = ("actually", "wrong", "myth", "lie", "truth is")
EMOTION_WORDS = {
    "love", "hate", "crazy", "insane", "shocked", "amazing", "terrible",
    "incredible", "stupid", "brilliant", "worst", "best", "unbelievable",
    "wild", "painful", "scared", "excited", "angry", "furious",
}
POSITIVE_WORDS = {
    "love", "amazing", "incredible", "brilliant", "best", "great",
    "awesome", "excited", "win", "success", "happy",
}
NEGATIVE_WORDS = {
    "hate", "terrible", "worst", "stupid", "painful", "fear", "scared",
    "angry", "fail", "lose", "broke", "sad",
}
ACTION_VERBS = {
    "build", "make", "stop", "start", "avoid", "create", "learn",
    "get", "grow", "fix", "quit", "launch", "scale",
}
CURIOSITY_PHRASES = (
    "what happened next",
    "here is why",
    "here's why",
    "the truth about",
    "but then",
    "until one day",
    "imagine",
    "what if",
)
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "is", "are", "was", "were", "be", "been", "it",
    "this", "that", "these", "those", "i", "you", "he", "she", "we",
    "they", "my", "your", "his", "her", "our", "their", "as", "so",
    "if", "not", "no", "do", "does", "did", "have", "has", "had",
    "will", "would", "can", "could", "just", "very", "really", "about",
    "from", "into", "over", "after", "before", "when", "where", "what",
    "which", "who", "how", "all", "any", "some", "more", "most", "than",
    "then", "there", "here", "out", "up", "down", "like", "also", "its",
    "it’s", "don", "didn", "doesn", "isn", "wasn", "aren", "weren",
}
FALLBACK_TAGS = ["#shorts", "#clips", "#viral", "#podcast", "#learn"]

SIGNAL_WORDS = {
    "hook": "a strong opening hook",
    "emotion": "high emotional energy",
    "density": "dense, specific detail",
    "pacing": "fast, energetic pacing",
    "curiosity": "a real curiosity gap",
}


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9']+", text.lower())


def _overlapping(segments: list[Segment], start: float, end: float) -> list[Segment]:
    return [s for s in segments if s.end > start and s.start < end]


def _window_text(segments: list[Segment], start: float, end: float) -> str:
    return " ".join(s.text.strip() for s in _overlapping(segments, start, end))


def _sentence_end_times(transcript: Transcript) -> list[float]:
    ends = []
    for seg in transcript.segments:
        stripped = seg.text.strip()
        if stripped and stripped[-1] in ".?!":
            ends.append(seg.end)
    return sorted(ends)


def _snap_edges(
    start: float, end: float, sentence_ends: list[float], duration: float
) -> tuple[float, float]:
    """Snap window edges to the nearest sentence end so cuts land on pauses."""
    past = [t for t in sentence_ends if t <= start]
    if past and start - past[-1] <= 10.0:
        start = past[-1]
    future = [t for t in sentence_ends if t >= end]
    if future and future[0] - end <= 10.0:
        end = future[0]
    start = max(0.0, min(start, duration))
    end = max(0.0, min(end, duration))
    if end - start < 5.0:
        end = min(duration, start + 5.0)
    return start, end


def _hook_score(opening_text: str) -> float:
    lowered = opening_text.lower()
    score = 0.0
    if "?" in opening_text:
        score += 0.35
    if re.search(r"\d", opening_text):
        score += 0.25
    if any(w in lowered for w in HOOK_WORDS):
        score += 0.25
    if any(w in lowered for w in CONTRARIAN_WORDS):
        score += 0.15
    return min(1.0, score)


def _emotion_score(text: str) -> float:
    toks = _tokens(text)
    if not toks:
        return 0.0
    score = 0.0
    score += min(0.30, text.count("!") * 0.10)
    caps = sum(1 for w in text.split() if len(w) > 1 and w.isupper())
    score += min(0.20, (caps / len(toks)) * 2.0)
    emo = sum(1 for w in toks if w in EMOTION_WORDS)
    score += min(0.30, (emo / len(toks)) * 3.0)
    pos = sum(1 for w in toks if w in POSITIVE_WORDS)
    neg = sum(1 for w in toks if w in NEGATIVE_WORDS)
    score += min(0.20, (abs(pos - neg) / len(toks)) * 2.0)
    return min(1.0, score)


def _density_score(text: str) -> float:
    toks = _tokens(text)
    if not toks:
        return 0.0
    score = 0.0
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    cap_count = 0
    for sentence in sentences:
        words = sentence.split()
        for word in words[1:]:
            if word[:1].isupper() and word.lower() not in STOPWORDS:
                cap_count += 1
    score += min(0.40, (cap_count / len(toks)) * 4.0)
    numbers = len(re.findall(r"\$\d[\d,.]*|\d+(?:\.\d+)?%?", text))
    score += min(0.30, numbers * 0.15)
    actions = sum(1 for w in toks if w in ACTION_VERBS)
    score += min(0.30, (actions / len(toks)) * 4.0)
    return min(1.0, score)


def _pacing_score(wps: float, mean_wps: float) -> float:
    if mean_wps <= 0:
        return 0.0
    ratio = wps / mean_wps
    return max(0.0, min(1.0, (ratio - 0.7) / 0.8))


def _curiosity_score(text: str) -> float:
    lowered = text.lower()
    hits = sum(1 for phrase in CURIOSITY_PHRASES if phrase in lowered)
    score = 0.40 * hits
    if "?" in text:
        score += 0.20
    return min(1.0, score)


def _first_sentence(text: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return parts[0].strip() if parts and parts[0].strip() else text.strip()[:80]


def _title(hook_line: str, limit: int = 60) -> str:
    text = hook_line.strip()
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut if cut else text[:limit]


def _hashtags(text: str, limit: int = 5) -> list[str]:
    toks = [w for w in _tokens(text) if w not in STOPWORDS and len(w) > 3]
    tags = ["#" + w for w, _ in Counter(toks).most_common(limit)]
    for fallback in FALLBACK_TAGS:
        if len(tags) >= 3:
            break
        if fallback not in tags:
            tags.append(fallback)
    return tags[:limit]


def _reason(sub_scores: dict[str, float]) -> str:
    ranked = sorted(sub_scores, key=lambda k: WEIGHTS[k] * sub_scores[k], reverse=True)
    first, second = SIGNAL_WORDS[ranked[0]], SIGNAL_WORDS[ranked[1]]
    return f"This moment scores well on {first} and {second}."


def score_transcript(transcript: Transcript) -> list[Candidate]:
    """Score sliding windows over the transcript. Returns candidates, best first."""
    segments = sorted(transcript.segments, key=lambda s: s.start)
    duration = transcript.duration
    if not segments or duration <= 0:
        return []

    sentence_ends = _sentence_end_times(transcript)
    all_text = " ".join(s.text for s in segments)
    all_tokens = _tokens(all_text)
    mean_wps = len(all_tokens) / duration if duration > 0 else 0.0

    candidates: list[Candidate] = []
    for length in WINDOW_LENGTHS:
        t = 0.0
        while t + MIN_WINDOW <= duration:
            raw_end = min(t + length, duration)
            start, end = _snap_edges(t, raw_end, sentence_ends, duration)
            if end - start >= MIN_WINDOW:
                candidates.append(
                    _score_window(transcript, segments, start, end, mean_wps)
                )
            t += WINDOW_STEP

    candidates.sort(key=lambda c: c.score, reverse=True)
    for i, candidate in enumerate(candidates, start=1):
        candidate.id = i
    return candidates


def _score_window(
    transcript: Transcript,
    segments: list[Segment],
    start: float,
    end: float,
    mean_wps: float,
) -> Candidate:
    text = _window_text(segments, start, end)
    opening = _window_text(segments, start, start + 3.0)
    toks = _tokens(text)
    wps = len(toks) / max(0.1, end - start)

    sub_scores = {
        "hook": _hook_score(opening),
        "emotion": _emotion_score(text),
        "density": _density_score(text),
        "pacing": _pacing_score(wps, mean_wps),
        "curiosity": _curiosity_score(text),
    }
    total = round(sum(WEIGHTS[k] * v for k, v in sub_scores.items()) * 100.0, 1)

    hook_line = _first_sentence(text)
    window_segments = _overlapping(segments, start, end)
    speakers = [s.speaker for s in window_segments if s.speaker]
    speaker = Counter(speakers).most_common(1)[0][0] if speakers else None

    return Candidate(
        id=0,
        start=round(start, 2),
        end=round(end, 2),
        score=total,
        hook_line=hook_line,
        reason=_reason(sub_scores),
        title=_title(hook_line),
        hashtags=_hashtags(text),
        speaker=speaker,
    )
