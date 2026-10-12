"""Groq API client for transcription and candidate ranking.

Transcription and LLM ranking are the two pipeline steps that need a
GROQ_API_KEY. Everything else in the backend runs locally.
"""
from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path

from app.core.config import settings
from app.pipeline import Candidate, Segment, Transcript, Word, pick_top

WHISPER_MODEL = "whisper-large-v3-turbo"
RANK_MODEL = "llama-3.3-70b-versatile"

# 10 minutes of 16 kHz mono 16-bit audio is about 19 MB, safely under the
# 25 MB per-request limit of the Groq transcription API.
CHUNK_SECONDS = 600
PROMPT_CHUNK_CHARS = 6000
PROMPT_OVERLAP_CHARS = 600


class GroqError(RuntimeError):
    """Raised when a Groq call fails or the API key is missing."""


def _require_key() -> str:
    key = settings.GROQ_API_KEY.strip()
    if not key:
        raise GroqError(
            "GROQ_API_KEY is not set. Transcription and LLM ranking need it. "
            "Get a free key at console.groq.com, put it in backend/.env, "
            "then restart the server."
        )
    return key


def _client():  # type: ignore[no-untyped-def]
    from groq import Groq

    return Groq(api_key=_require_key())


def _probe_duration(path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return 0.0


def extract_audio_wav(source: Path, out_dir: Path) -> Path:
    """Extract the source audio to 16 kHz mono wav for transcription."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "audio_16k.wav"
    cmd = [
        "ffmpeg", "-y", "-v", "error", "-i", str(source),
        "-vn", "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le", str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not out.exists():
        raise GroqError(f"ffmpeg audio extraction failed: {proc.stderr[-2000:]}")
    return out


def _split_wav(wav: Path, tmpdir: Path) -> list[tuple[Path, float]]:
    """Split a wav into CHUNK_SECONDS pieces. Returns (path, offset) pairs."""
    duration = _probe_duration(wav)
    if duration <= 0:
        return [(wav, 0.0)]
    parts: list[tuple[Path, float]] = []
    offset = 0.0
    index = 0
    while offset < duration:
        part = tmpdir / f"part_{index:03d}.wav"
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-ss", f"{offset:.2f}", "-t", str(CHUNK_SECONDS),
            "-i", str(wav), "-acodec", "pcm_s16le", str(part),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not part.exists():
            raise GroqError(f"ffmpeg audio chunking failed: {proc.stderr[-2000:]}")
        parts.append((part, offset))
        offset += CHUNK_SECONDS
        index += 1
    return parts


def _transcribe_chunk(client, wav_path: Path) -> dict:  # type: ignore[no-untyped-def]
    with open(wav_path, "rb") as fh:
        resp = client.audio.transcriptions.create(
            file=(wav_path.name, fh),
            model=WHISPER_MODEL,
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"],
            language="en",
        )
    if hasattr(resp, "model_dump"):
        return resp.model_dump()
    if hasattr(resp, "to_dict"):
        return resp.to_dict()
    return dict(resp)


def _map_transcript(
    chunks: list[tuple[dict, float]],
    video_id: str,
    duration: float,
) -> Transcript:
    segments: list[Segment] = []
    for data, offset in chunks:
        for seg in data.get("segments", []):
            words: list[Word] = []
            for w in seg.get("words") or []:
                text = str(w.get("word", "") or w.get("text", "")).strip()
                if not text:
                    continue
                words.append(
                    Word(
                        text=text,
                        start=float(w.get("start", 0.0)) + offset,
                        end=float(w.get("end", 0.0)) + offset,
                    )
                )
            text = str(seg.get("text", "")).strip()
            if not text:
                continue
            segments.append(
                Segment(
                    id=len(segments),
                    start=float(seg.get("start", 0.0)) + offset,
                    end=float(seg.get("end", 0.0)) + offset,
                    text=text,
                    words=words,
                )
            )
    if duration <= 0 and segments:
        duration = segments[-1].end
    return Transcript(video_id=video_id, duration=duration, segments=segments)


def transcribe(audio_path: Path, video_id: str) -> Transcript:
    """Transcribe a video file with Groq Whisper.

    The audio is extracted to 16 kHz wav first, then sent in 10 minute
    chunks so long videos stay under the API size limit. Returns a
    Transcript with segment and word level timestamps.
    """
    client = _client()
    with tempfile.TemporaryDirectory(prefix="clipforge_") as tmp:
        tmpdir = Path(tmp)
        wav = extract_audio_wav(Path(audio_path), tmpdir)
        duration = _probe_duration(Path(audio_path))
        parts = _split_wav(wav, tmpdir)
        chunks: list[tuple[dict, float]] = []
        for part_path, offset in parts:
            chunks.append((_transcribe_chunk(client, part_path), offset))
    return _map_transcript(chunks, video_id, duration)


def _segments_to_text(segments: list[Segment]) -> str:
    lines = []
    for s in segments:
        speaker = f"[{s.speaker}] " if s.speaker else ""
        lines.append(f"[{s.start:.1f}-{s.end:.1f}] {speaker}{s.text}")
    return "\n".join(lines)


def _chunk_segments(segments: list[Segment]) -> list[list[Segment]]:
    """Group segments into prompt sized chunks with a small overlap."""
    chunks: list[list[Segment]] = []
    current: list[Segment] = []
    current_chars = 0
    for seg in segments:
        text = _segments_to_text([seg])
        if current and current_chars + len(text) > PROMPT_CHUNK_CHARS:
            chunks.append(current)
            overlap = _segments_to_text(current)[-PROMPT_OVERLAP_CHARS:]
            current = [s for s in current if overlap in _segments_to_text([s])]
            current_chars = sum(len(_segments_to_text([s])) for s in current)
        current.append(seg)
        current_chars += len(text)
    if current:
        chunks.append(current)
    return chunks


_SYSTEM_PROMPT = (
    "You are a short-form video editor. Given a transcript chunk and a list of "
    "heuristic clip candidates, pick the moments most likely to perform as "
    "TikTok, Reels, or YouTube Shorts. Favor strong hooks in the first 3 "
    "seconds, emotional peaks, surprising claims, concrete advice, and quotable "
    "lines. Return ONLY a JSON object of the form "
    '{"candidates": [{"start": 12.5, "end": 48.0, "score": 87, '
    '"hook_line": "...", "reason": "...", "title": "...", '
    '"hashtags": ["#tag1", "#tag2"]}]}. '
    "Scores are 0 to 100. start and end are seconds. hook_line is the exact "
    "opening line of the clip. Keep titles under 60 characters."
)


def _strip_fences(text: str) -> str:
    text = text.strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    return match.group(1) if match else text


def _parse_llm_json(text: str) -> list[dict]:
    data = json.loads(_strip_fences(text))
    if isinstance(data, dict) and "candidates" in data:
        data = data["candidates"]
    if not isinstance(data, list):
        raise ValueError("LLM response was not a JSON array")
    return [d for d in data if isinstance(d, dict)]


def _snap_end(end: float, segments: list[Segment]) -> float:
    """Snap a clip end to the nearest sentence end within a small window."""
    best = end
    best_dist = 3.0
    for seg in segments:
        dist = abs(seg.end - end)
        if dist < best_dist:
            best = seg.end
            best_dist = dist
    return best


def _rank_chunk(
    client,  # type: ignore[no-untyped-def]
    segments: list[Segment],
    heuristic: list[Candidate],
    duration: float,
) -> list[Candidate]:
    chunk_text = _segments_to_text(segments)
    heur_text = "\n".join(
        f"- [{c.start:.1f}-{c.end:.1f}] score={c.score:.0f}: {c.title or c.hook_line}"
        for c in heuristic
    )
    user_prompt = (
        f"Transcript chunk (timestamps in seconds):\n{chunk_text}\n\n"
        f"Heuristic candidates in this range:\n{heur_text or '(none)'}\n\n"
        "Rank the best moments. Return the JSON object now."
    )
    last_error: Exception | None = None
    items: list[dict] = []
    for _attempt in range(2):
        try:
            resp = client.chat.completions.create(
                model=RANK_MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content or ""
            items = _parse_llm_json(raw)
            last_error = None
            break
        except Exception as exc:  # retry once on parse or API failure
            last_error = exc
    if last_error is not None:
        raise GroqError(f"LLM ranking failed twice: {last_error}")

    out: list[Candidate] = []
    for item in items:
        try:
            start = max(0.0, min(float(item["start"]), duration))
            end = max(0.0, min(float(item["end"]), duration))
        except (KeyError, TypeError, ValueError):
            continue
        if end - start < 5:
            continue
        end = _snap_end(end, segments)
        hashtags = item.get("hashtags") or []
        out.append(
            Candidate(
                id=0,  # real ids are assigned by pick_top after merging
                start=start,
                end=end,
                score=max(0.0, min(100.0, float(item.get("score", 50)))),
                hook_line=str(item.get("hook_line", "") or "")[:200],
                reason=str(item.get("reason", "") or "")[:300],
                title=str(item.get("title", "") or "")[:80],
                hashtags=[str(h) for h in hashtags][:8],
            )
        )
    return out


def _overlap(a_start: float, a_end: float, b: Candidate) -> float:
    inter = max(0.0, min(a_end, b.end) - max(a_start, b.start))
    union = max(a_end, b.end) - min(a_start, b.start)
    return inter / union if union > 0 else 0.0


def rank_candidates(
    transcript: Transcript,
    heuristic_candidates: list[Candidate],
) -> list[Candidate]:
    """Refine heuristic candidates with LLM scoring.

    The transcript is sent in chunks. LLM scores win; speaker labels and any
    missing fields are filled from the overlapping heuristic candidate. The
    result is deduped with pick_top, given fresh integer ids, and sorted by
    score, highest first.
    """
    client = _client()
    duration = transcript.duration
    ranked: list[Candidate] = []
    chunks = _chunk_segments(transcript.segments)
    for chunk in chunks:
        if not chunk:
            continue
        lo, hi = chunk[0].start, chunk[-1].end
        heur = [c for c in heuristic_candidates if c.end >= lo and c.start <= hi]
        try:
            ranked.extend(_rank_chunk(client, chunk, heur, duration))
        except GroqError:
            ranked.extend(heur)

    merged: list[Candidate] = []
    for cand in ranked:
        best: Candidate | None = None
        best_ov = 0.0
        for h in heuristic_candidates:
            ov = _overlap(cand.start, cand.end, h)
            if ov > best_ov:
                best, best_ov = h, ov
        if best is not None and best_ov > 0.3:
            if not cand.speaker:
                cand.speaker = best.speaker
            if not cand.title and best.title:
                cand.title = best.title
            if not cand.hook_line and best.hook_line:
                cand.hook_line = best.hook_line
        merged.append(cand)

    # Top up with strong heuristic candidates the LLM missed.
    for h in sorted(heuristic_candidates, key=lambda c: c.score, reverse=True):
        if all(_overlap(h.start, h.end, m) < 0.5 for m in merged):
            merged.append(h)

    final = pick_top(merged, n=len(merged))
    final.sort(key=lambda c: c.score, reverse=True)
    return final
