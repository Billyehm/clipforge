"""Pipeline orchestration: transcribe -> score -> rank -> render.

Each stage is cached with BIT 1's StageCache under work/<video_id>/, so a
re-run skips finished stages unless fresh=True is passed. Progress is
reported through work/<video_id>/progress.json for the frontend to poll.
"""
from __future__ import annotations

import json
from pathlib import Path

from app.core.config import work_dir
from app.pipeline import (
    Candidate,
    ClipRecord,
    StageCache,
    Transcript,
    pick_top,
    score_transcript,
)
from app.services.groq import GroqError, transcribe
from app.services.render import _find_source, render_clip

STAGE_NAMES = ("transcript", "candidates", "clips")


def video_dir(video_id: str) -> Path:
    return work_dir() / video_id


def stage_cache(video_id: str) -> StageCache:
    return StageCache(video_id, root=work_dir())


def set_progress(
    video_id: str,
    stage: str,
    done: int = 0,
    total: int = 0,
    error: str | None = None,
) -> None:
    path = video_dir(video_id) / "progress.json"
    path.write_text(
        json.dumps({"stage": stage, "done": done, "total": total, "error": error})
    )


def read_progress(video_id: str) -> dict:
    path = video_dir(video_id) / "progress.json"
    if not path.exists():
        return {"stage": "idle", "done": 0, "total": 0, "error": None}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {"stage": "idle", "done": 0, "total": 0, "error": None}


def _naive_candidates(transcript: Transcript, n: int) -> list[Candidate]:
    """Last-resort scorer if BIT 1's score_transcript ever fails to import.

    Slides a 30 second window across the transcript and scores each window
    on word density plus hook signals. The LLM rank step still refines the
    ordering afterwards.
    """
    window, step = 30.0, 10.0
    duration = transcript.duration
    if duration <= 0:
        return []
    cands: list[Candidate] = []
    t = 0.0
    idx = 1
    while t < duration:
        end = min(t + window, duration)
        segs = [s for s in transcript.segments if s.end > t and s.start < end]
        text = " ".join(s.text for s in segs)
        wc = len(text.split())
        score = min(70.0, wc * 0.6)
        if "?" in text:
            score += 10
        if "!" in text:
            score += 8
        if any(ch.isdigit() for ch in text):
            score += 6
        hook = segs[0].text.strip()[:120] if segs else ""
        cands.append(
            Candidate(
                id=idx,
                start=t,
                end=end,
                score=round(min(95.0, score), 1),
                hook_line=hook,
                reason="Heuristic pick: dense speech with hook signals.",
                title=(segs[0].text.strip()[:60] if segs else f"Moment {idx}"),
                hashtags=[],
            )
        )
        t += step
        idx += 1
    cands.sort(key=lambda c: c.score, reverse=True)
    picked: list[Candidate] = []
    for c in cands:
        if all(min(c.end, p.end) - max(c.start, p.start) < 15.0 for p in picked):
            picked.append(c)
        if len(picked) >= n * 2:
            break
    return pick_top(picked, n=len(picked)) if picked else []


def _heuristic_candidates(transcript: Transcript, n: int) -> list[Candidate]:
    """Score with BIT 1's scorer and keep twice what we need for the LLM."""
    try:
        scored = score_transcript(transcript)
        return pick_top(scored, n=n * 2)
    except Exception:
        return _naive_candidates(transcript, n)


def _edl_entry(clip: ClipRecord) -> dict:
    """One edit-decision-list entry describing exactly how a clip was made."""
    c = clip.candidate
    return {
        "clip_id": clip.id,
        "title": c.title,
        "description": c.description,
        "genre": c.genre,
        "hook_line": c.hook_line,
        "score": c.score,
        "source_start": round(c.start, 2),
        "source_end": round(c.end, 2),
        "duration": round(c.end - c.start, 2),
        "output": clip.output_path,
        "captions_ass": clip.caption_path,
        "captions_srt": clip.srt_path,
        "hashtags": c.hashtags,
        "thumbnail_concept": c.thumbnail_concept,
        "render": clip.render_info,
        "status": clip.status,
        "error": clip.error,
    }


def run_pipeline(video_id: str, n: int = 8, fresh: bool = False) -> None:
    """Run the full pipeline in a background task. Never raises to the caller."""
    vdir = video_dir(video_id)
    if not vdir.exists():
        return
    cache = stage_cache(video_id)
    try:
        # Stage 1: transcribe
        set_progress(video_id, "transcribe", 0, 1)
        data = None if fresh else cache.load("transcript")
        if data:
            transcript = Transcript.model_validate(data)
        else:
            source = _find_source(vdir)

            def _on_chunk(done: int, total: int) -> None:
                set_progress(video_id, "transcribe", done, total)

            transcript = transcribe(source, video_id, on_progress=_on_chunk)
            cache.save("transcript", transcript.model_dump())
        set_progress(video_id, "transcribe", 1, 1)

        # Stage 2: heuristic score and select
        set_progress(video_id, "score", 0, 1)
        data = None if fresh else cache.load("candidates")
        if data:
            candidates = [Candidate.model_validate(c) for c in data]
        else:
            candidates = _heuristic_candidates(transcript, n)
            cache.save("candidates", [c.model_dump() for c in candidates])
        set_progress(video_id, "score", 1, 1)

        # Stage 3: LLM rank refinement
        set_progress(video_id, "rank", 0, 1)
        try:
            from app.services.groq import rank_candidates

            candidates = rank_candidates(transcript, candidates)
            cache.save("candidates", [c.model_dump() for c in candidates])
        except GroqError as exc:
            # Keep heuristic candidates; the pipeline still produces clips.
            set_progress(video_id, "rank", 0, 1, error=str(exc)[:500])
        set_progress(video_id, "rank", 1, 1)

        # Stage 4: final top-N selection and clip records
        top = pick_top(candidates, n=n)
        clips = [
            ClipRecord(id=f"clip_{i:03d}", video_id=video_id, candidate=c)
            for i, c in enumerate(top)
        ]
        cache.save("clips", [c.model_dump() for c in clips])

        # Stage 5: render
        total = len(clips)
        for i, clip in enumerate(clips):
            set_progress(video_id, "render", i, total)
            render_clip(video_id, clip, work_dir())
            clips[i] = clip
            cache.save("clips", [c.model_dump() for c in clips])
        set_progress(video_id, "render", total, total)

        # Stage 6: edit decision list, one entry per clip.
        cache.save("edl", [_edl_entry(c) for c in clips])

        set_progress(video_id, "done", total, total)
    except Exception as exc:
        set_progress(video_id, "error", 0, 0, error=str(exc)[:2000])
