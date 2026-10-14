"""FFmpeg clip rendering: cut, reframe to 9:16, burn captions.

Two render paths:
- Advanced (cv2 available): frames are decoded with OpenCV, re-framed
  per frame (single face-tracked crop with a gentle zoom pulse, or a
  two-speaker stacked split-screen), then vignette and a gold progress
  bar are applied. Frames are piped to ffmpeg, which burns the ASS
  captions and muxes the original audio.
- Basic (cv2 missing): the original single ffmpeg command with a static
  center crop.

Caption building is shared: karaoke ASS from word timestamps (shifted
to clip-relative time), with the entrance effect from settings, falling
back to plain segment subtitles when words are missing.
"""
from __future__ import annotations

import glob
import subprocess
from pathlib import Path

from app.pipeline import (
    StageCache,
    Transcript,
    Word,
    detect_genre,
    get_profile,
    words_to_ass,
    words_to_srt,
)
from app.pipeline.models import ClipRecord
from app.core.config import settings, work_dir

CLIP_W = 1080
CLIP_H = 1920
OUT_FPS = 30
GOLD_BGR = (55, 175, 212)  # #d4af37


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


def _resolve_effect(profile: dict) -> str:
    """Caption effect: explicit setting wins, otherwise the genre profile."""
    if settings.CAPTION_EFFECT != "auto":
        return settings.CAPTION_EFFECT
    return profile.get("caption_effect", "pop")


def _build_ass(
    clip: ClipRecord,
    transcript: Transcript | None,
    ass_path: Path,
    profile: dict,
) -> None:
    """Write karaoke ASS captions, falling back to plain subtitles."""
    words = _clip_words(clip, transcript)
    if words:
        try:
            content = words_to_ass(
                words,
                effect=_resolve_effect(profile),
                accent=profile.get("accent", "yellow"),
            )
            ass_path.parent.mkdir(parents=True, exist_ok=True)
            ass_path.write_text(content, encoding="utf-8")
            return
        except Exception:
            pass
    _fallback_ass(clip, transcript, ass_path)


def _build_srt(
    clip: ClipRecord,
    transcript: Transcript | None,
    srt_path: Path,
) -> bool:
    """Write a plain SRT caption file. Returns True when words existed."""
    words = _clip_words(clip, transcript)
    if not words:
        return False
    try:
        srt_path.parent.mkdir(parents=True, exist_ok=True)
        srt_path.write_text(words_to_srt(words), encoding="utf-8")
        return True
    except Exception:
        return False


def _crop_filter(clip: ClipRecord, source: Path) -> str:
    """Face-tracked crop when possible, static center crop otherwise.

    plan_reframe returns per-sample face centers; the mean center becomes a
    static crop position for this render. Used only by the basic path.
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


def _vignette_mask() -> "object":
    """Radial darkening mask: 1.0 in the center, about 0.82 at corners."""
    import numpy as np

    yy, xx = np.mgrid[0:CLIP_H, 0:CLIP_W].astype(np.float32)
    dx = (xx - CLIP_W / 2) / (CLIP_W / 2)
    dy = (yy - CLIP_H / 2) / (CLIP_H / 2)
    r = np.sqrt(dx * dx + dy * dy) / 1.4142
    mask = 1.0 - 0.18 * np.clip(r, 0.0, 1.0)
    return mask[:, :, None].astype(np.float32)


def _zoom(t: float, dur: float, intensity: float = 0.07) -> float:
    """Gentle zoom pulse that tightens over the life of the clip."""
    return 1.0 + intensity * (t / dur if dur > 0 else 0.0)


def _single_crop(
    frame: "object", track: list, t: float, dur: float, zoom_intensity: float
) -> "object":
    """9:16 crop following one face track, with a zoom pulse."""
    import cv2
    import numpy as np

    from app.pipeline.reframe import _interp_track

    fh, fw = frame.shape[:2]
    cx, cy, _s = _interp_track(track, t)
    z = _zoom(t, dur, zoom_intensity)
    cw = (fh * 9.0 / 16.0) / z
    ch = fh / z
    if cw > fw:
        # Vertical source: static center crop fitting 9:16 inside the frame.
        cw = min(float(fw), fh * 9.0 / 16.0)
        ch = cw * 16.0 / 9.0
        fx, fy = fw / 2.0, fh / 2.0
    else:
        fx = min(max(cx * fw, cw / 2), fw - cw / 2)
        fy = min(max(cy * fh, ch / 2), fh - ch / 2)
    x0 = int(round(fx - cw / 2))
    y0 = int(round(fy - ch / 2))
    x1 = x0 + int(round(cw))
    y1 = y0 + int(round(ch))
    x0 = max(0, min(x0, fw - 1))
    y0 = max(0, min(y0, fh - 1))
    x1 = max(x0 + 1, min(x1, fw))
    y1 = max(y0 + 1, min(y1, fh))
    crop = frame[y0:y1, x0:x1]
    return cv2.resize(crop, (CLIP_W, CLIP_H), interpolation=cv2.INTER_LINEAR)


def _split_half(
    frame: "object", track: list, t: float, dur: float, zoom_intensity: float
) -> "object":
    """One 1080x960 half of the split-screen, framing one speaker."""
    import cv2

    from app.pipeline.reframe import _interp_track

    fh, fw = frame.shape[:2]
    if len(track) >= 2:
        cx, cy, s = _interp_track(track, t)
    else:
        cx, cy, s = 0.5, 0.5, 0.3
    z = _zoom(t, dur, zoom_intensity)
    side = s * fw * 5.0 / z
    side = min(max(side, fh * 0.35), fh * 0.95)
    # Box sits slightly above the face center so there is headroom below.
    bx = min(max(cx * fw, side / 2), fw - side / 2)
    by = min(max((cy - 0.06) * fh, side / 2), fh - side / 2)
    x0 = int(round(bx - side / 2))
    y0 = int(round(by - side / 2))
    x1 = x0 + int(round(side))
    y1 = y0 + int(round(side))
    x0 = max(0, min(x0, fw - 1))
    y0 = max(0, min(y0, fh - 1))
    x1 = max(x0 + 1, min(x1, fw))
    y1 = max(y0 + 1, min(y1, fh))
    crop = frame[y0:y1, x0:x1]
    return cv2.resize(crop, (CLIP_W, CLIP_H // 2), interpolation=cv2.INTER_LINEAR)


def _process_frame(
    frame: "object",
    plan: dict,
    t: float,
    dur: float,
    vignette: "object",
    profile: dict,
) -> "object":
    """Build one finished 1080x1920 frame: reframe, vignette, progress."""
    import numpy as np

    zoom_intensity = float(profile.get("zoom", 0.07))
    mode = plan.get("mode", "center")
    tracks = plan.get("tracks", [])
    if mode == "split" and profile.get("allow_split", True) and len(tracks) >= 2:
        # Top half: most present speaker; bottom half: second speaker.
        # A track with too few samples reuses the other track's region
        # so a half is never left black.
        usable = [len(tr) >= 2 for tr in tracks[:2]]
        halves = []
        for idx in range(2):
            src_track = tracks[idx] if usable[idx] else tracks[1 - idx]
            halves.append(_split_half(frame, src_track, t, dur, zoom_intensity))
        out = np.zeros((CLIP_H, CLIP_W, 3), dtype=np.uint8)
        out[0 : CLIP_H // 2] = halves[0]
        out[CLIP_H // 2 : CLIP_H] = halves[1]
    else:
        track = tracks[0] if tracks else []
        out = _single_crop(frame, track, t, dur, zoom_intensity)

    out = np.clip(
        out.astype(np.float32) * vignette, 0, 255
    ).astype(np.uint8)

    width = int(CLIP_W * min(1.0, max(0.0, t / dur if dur > 0 else 0.0)))
    if width > 0:
        out[0:10, 0:width] = GOLD_BGR
    return out


def _render_advanced(
    clip: ClipRecord,
    source: Path,
    ass_path: Path,
    out_path: Path,
    start: float,
    dur: float,
    plan: dict,
    profile: dict,
) -> None:
    """Per-frame edited render piped to ffmpeg.

    Frames are decoded with OpenCV, reframed per frame, finished with a
    vignette and progress bar, then piped as raw BGR to ffmpeg, which
    burns the ASS captions and muxes the source audio.
    """
    import cv2

    n_frames = max(1, round(dur * OUT_FPS))
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RenderError(f"Could not open video: {source}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * src_fps))

    vignette = _vignette_mask()
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{CLIP_W}x{CLIP_H}", "-framerate", str(OUT_FPS),
        "-i", "pipe:0",
        "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(source),
        "-map", "0:v", "-map", "1:a",
        "-vf", f"subtitles={_escape_subtitles_path(ass_path)}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart", "-shortest",
        str(out_path),
    ]
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE
    )
    try:
        last = None
        for i in range(n_frames):
            t = i / OUT_FPS
            ok, frame = cap.read()
            if ok and frame is not None:
                last = frame
            if last is None:
                raise RenderError("Could not read any frame from source video")
            out = _process_frame(last, plan, t, dur, vignette, profile)
            try:
                proc.stdin.write(out.tobytes())
            except BrokenPipeError:
                break
    finally:
        cap.release()
        try:
            proc.stdin.close()
        except BrokenPipeError:
            pass
    stderr = proc.stderr.read().decode("utf-8", "replace")
    rc = proc.wait()
    if rc != 0 or not out_path.exists():
        tail = stderr.strip().splitlines()[-15:]
        raise RenderError(
            f"ffmpeg failed for clip {clip.id}:\n" + "\n".join(tail)
        )


def _render_basic(
    clip: ClipRecord,
    source: Path,
    ass_path: Path,
    out_path: Path,
    start: float,
    dur: float,
) -> None:
    """Original single-ffmpeg-command render with a static center crop."""
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


def render_clip(video_id: str, clip: ClipRecord, work_dir: str | Path) -> ClipRecord:
    """Render one clip: cut, reframe to 9:16, burn captions, encode.

    Updates the ClipRecord with output_path and caption_path and sets status
    to "done", or to "failed" with the error on failure.
    """
    video_dir = Path(work_dir) / video_id
    try:
        source = _find_source(video_dir)
        transcript = _load_transcript(video_id)

        genre = clip.candidate.genre or (
            detect_genre(transcript) if transcript else "podcast"
        )
        profile = get_profile(genre)

        captions_dir = video_dir / "captions"
        clips_dir = video_dir / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)
        ass_path = captions_dir / f"{clip.id}.ass"
        srt_path = captions_dir / f"{clip.id}.srt"
        out_path = clips_dir / f"{clip.id}.mp4"

        _build_ass(clip, transcript, ass_path, profile)
        srt_ok = _build_srt(clip, transcript, srt_path)

        start = max(0.0, clip.candidate.start)
        dur = clip.candidate.end - clip.candidate.start
        if dur < 0.5:
            raise RenderError(
                f"Clip {clip.id} is too short ({dur:.2f}s), need at least 0.5s"
            )

        plan = None
        try:
            import cv2  # noqa: F401
        except ImportError:
            plan = None
        else:
            from app.pipeline.reframe import plan_reframe

            try:
                plan = plan_reframe(str(source), start, start + dur)
            except RuntimeError:
                # Missing mediapipe or unreadable video: basic path instead.
                plan = None

        if plan is None:
            _render_basic(clip, source, ass_path, out_path, start, dur)
            render_mode = "basic"
        else:
            _render_advanced(clip, source, ass_path, out_path, start, dur, plan, profile)
            render_mode = plan.get("mode", "track")

        clip.status = "done"
        clip.output_path = str(out_path)
        clip.caption_path = str(ass_path)
        clip.srt_path = str(srt_path) if srt_ok else None
        clip.render_info = {
            "genre": genre,
            "profile": profile.get("label", genre),
            "mode": render_mode,
            "caption_effect": _resolve_effect(profile),
            "caption_accent": profile.get("accent", "yellow"),
            "zoom": profile.get("zoom", 0.07),
            "vignette": render_mode != "basic",
            "progress_bar": render_mode != "basic",
        }
        clip.error = None
    except Exception as exc:
        clip.status = "failed"
        clip.error = str(exc)[:2000]
    return clip
