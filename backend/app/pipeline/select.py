"""Turn scored candidates into the final clip list.

Overlapping candidates are collapsed so the same moment never
produces two clips, then the top N survive with fresh sequential ids.
"""

from .models import Candidate


def _overlap_ratio(a: Candidate, b: Candidate) -> float:
    """Intersection over the shorter candidate's duration, 0 to 1."""
    inter = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    shorter = min(a.end - a.start, b.end - b.start)
    if shorter <= 0:
        return 0.0
    return inter / shorter


def dedupe(candidates: list[Candidate]) -> list[Candidate]:
    """Keep the higher score when two candidates overlap by over 50%."""
    ordered = sorted(candidates, key=lambda c: c.score, reverse=True)
    kept: list[Candidate] = []
    for candidate in ordered:
        if all(_overlap_ratio(candidate, other) <= 0.5 for other in kept):
            kept.append(candidate)
    return kept


def pick_top(
    candidates: list[Candidate],
    n: int = 10,
    min_len: float = 15,
    max_len: float = 90,
) -> list[Candidate]:
    """Filter by clip length, dedupe, and return the top n with fresh ids."""
    in_range = [
        c for c in candidates if min_len <= (c.end - c.start) <= max_len
    ]
    chosen = dedupe(in_range)[:n]
    return [c.model_copy(update={"id": i}) for i, c in enumerate(chosen, start=1)]
