"""FFmpeg clip rendering: cut, reframe to 9:16, burn captions.

Uses BIT 1's pipeline package directly:
- words_to_ass builds karaoke ASS captions from word timestamps.
- plan_reframe plans a face-tracked crop path per clip.
Both calls are guarded: if they fail (for example mediapipe is not
installed), the render falls back to plain segment subtitles and a static
center crop instead of failing the clip.
"""
from __future__ import annotations

import glob
import subprocess
from pathlib import Path

from app.pipeline import StageCache, Transcript, Word, words_to_ass
from app.pipeline.models import ClipRecord
from app.core.config import work_dir

CLIP_W = 1080
CLIP_H = 1920


class RenderError(RuntimeError):
    """Raised when ffmpeg fails to render a clip."""


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True)


def _find_source(video_dir: Path) -> Path:
    matches = sorted(glob.glob(str(video_dir / "source.*")))
    if not matches:
        raise RenderError(f"No source video found in {video_dir}")
    return Path(matches[0])


def _load_transcript(video_id: str) -> Transcript | None:
    cache = StageCache(video_id, root=work_dir())
    data = cache.load("transcript")
    if not data:
        return None
    try:
        return Transcript.model_validate(data)
    except Exception:
        return None


def _ass_timestamp(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int(round((seconds - int(seconds)) * 100))
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _fallback_ass(clip: ClipRecord, transcript: Transcript | None, ass_path: Path) -> None:
    """Plain segment subtitles used when karaoke captions cannot be built."""
    start, end = clip.candidate.start, clip.candidate.end
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {CLIP_W}\n"
        f"PlayResY: {CLIP_H}\n"
        "\n[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Alignment, MarginL, MarginR, MarginV, "
        "Outline, Shadow\n"
        "Style: Default,Arial,64,&H00FFFFFF,&H80000000,&H80000000,"
        "-1,0,2,60,60,420,3,1\n"
        "\n[Events]\n"
        "Format: Layer, Start, End, Style, Text\n"
    )
    lines = [header]
    if transcript:
        for seg in transcript.segments:
            if seg.end <= start or seg.start >= end:
                continue
            s = max(0.0, seg.start - start)
            e = max(0.0, seg.end - start)
            text = seg.text.replace("\n", " ").strip()
            if text:
                lines.append(
                    f"Dialogue: 0,{_ass_timestamp(s)},{_ass_timestamp(e)},Default,{text}"
                )
    ass_path.parent.mkdir(parents=True, exist_ok=True)
    ass_path.write_text("\n".join(lines), encoding="utf-8")


def _clip_words(clip: ClipRecord, transcript: Transcript | None) -> list[Word]:
    """Words inside the clip window, shifted so the clip starts at 0."""
    if not transcript:
        return []
    start, end = clip.candidate.start, clip.candidate.end
    words: list[Word] = []
    for seg in transcript.segments:
        for w in seg.words:
            if w.end <= start or w.start >= end:
                continue
            words.append(
                Word(
                    text=w.text,
                    start=max(0.0, w.start - start),
                    end=max(0.0, w.end - start),
                )
            )
    return words


def _build_ass(
    clip: ClipRecord,
    transcript: Transcript | None,
    ass_path: Path,
) -> None:
    """Write karaoke ASS captions, falling back to plain subtitles."""
    words = _clip_words(clip, transcript)
    if words:
        try:
            content = words_to_ass(words)
            ass_path.parent.mkdir(parents=True, exist_ok=True)
            ass_path.write_text(content, encoding="utf-8")
            return
        except Exception:
            pass
    _fallback_ass(clip, transcript, ass_path)


def _crop_filter(clip: ClipRecord, source: Path) -> str:
    """Face-tracked crop when possible, static center crop otherwise.

    plan_reframe returns per-sample face centers; the mean center becomes a
    static crop position for this render. A per-frame tracked crop is a
    possible future upgrade.
    """
    try:
        from app.pipeline import plan_reframe

        plan = plan_reframe(
            str(source), clip.candidate.start, clip.candidate.end
        )
        path = plan.get("path", [])
        if path:
            cx = sum(x for _, x in path) / len(path)
            cx = max(0.05, min(0.95, cx))
            return f"crop=ih*9/16:ih:(iw-ih*9/16)*{cx:.4f}"
    except Exception:
        pass
    return "crop=ih*9/16:ih"


def _escape_subtitles_path(path: Path) -> str:
    p = str(path).replace("\\", "/")
    for ch in (":", ",", "'", "[", "]", "="):
        p = p.replace(ch, "\\" + ch)
    return f"'{p}'"


def render_clip(video_id: str, clip: ClipRecord, work_dir: str | Path) -> ClipRecord:
    """Render one clip: cut, reframe to 9:16, burn captions, encode.

    Updates the ClipRecord with output_path and caption_path and sets status
    to "done", or to "failed" with the ffmpeg error tail on failure.
    """
    video_dir = Path(work_dir) / video_id
    try:
        source = _find_source(video_dir)
        transcript = _load_transcript(video_id)

        captions_dir = video_dir / "captions"
        clips_dir = video_dir / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)
        ass_path = captions_dir / f"{clip.id}.ass"
        out_path = clips_dir / f"{clip.id}.mp4"

        _build_ass(clip, transcript, ass_path)

        start = max(0.0, clip.candidate.start)
        dur = clip.candidate.end - clip.candidate.start
        if dur <= 0:
            raise RenderError(f"Clip {clip.id} has non-positive duration ({dur:.2f}s)")

        vf = (
            f"{_crop_filter(clip, source)},"
            f"scale={CLIP_W}:{CLIP_H},"
            f"setsar=1,"
            f"subtitles={_escape_subtitles_path(ass_path)},"
            f"format=yuv420p"
        )
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-ss", f"{start:.3f}", "-accurate_seek",
            "-i", str(source),
            "-t", f"{dur:.3f}",
            "-vf", vf,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            str(out_path),
        ]
        proc = _run(cmd)
        if proc.returncode != 0 or not out_path.exists():
            tail = (proc.stderr or "").strip().splitlines()[-15:]
            raise RenderError(f"ffmpeg failed for clip {clip.id}:\n" + "\n".join(tail))

        clip.status = "done"
        clip.output_path = str(out_path)
        clip.caption_path = str(ass_path)
        clip.error = None
    except Exception as exc:
        clip.status = "failed"
        clip.error = str(exc)[:2000]
    return clip
