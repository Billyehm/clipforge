"""HTTP API routes for the clipforge backend."""
from __future__ import annotations

import subprocess
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.core.config import settings, work_dir
from app.pipeline import Candidate, ClipRecord, StageCache
from app.services.media import trim_to_size
from app.services.pipeline import (
    STAGE_NAMES,
    read_progress,
    run_pipeline,
    set_progress,
    stage_cache,
    video_dir,
)
from app.services.render import render_clip

router = APIRouter()

ALLOWED_EXTS = {"mp4", "mov", "mkv", "webm"}


class RerenderRequest(BaseModel):
    start: float | None = None
    end: float | None = None
    title: str | None = None


def _require_video(video_id: str) -> Path:
    vdir = video_dir(video_id)
    if not stage_cache(video_id).exists("ingest"):
        raise HTTPException(404, f"Video {video_id} not found")
    return vdir


def _probe_duration(path: Path) -> float | None:
    try:
        proc = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return float(proc.stdout.strip())
    except Exception:
        return None


@router.post("/videos")
async def upload_video(file: UploadFile = File(...)) -> dict:
    """Accept a video upload and register it for processing."""
    ext = Path(file.filename or "").suffix.lower().lstrip(".")
    if ext not in ALLOWED_EXTS:
        raise HTTPException(
            400,
            f"Unsupported file type '.{ext}'. Use one of: {', '.join(sorted(ALLOWED_EXTS))}.",
        )
    video_id = uuid.uuid4().hex[:12]
    vdir = video_dir(video_id)
    vdir.mkdir(parents=True, exist_ok=True)
    dest = vdir / f"source.{ext}"
    tmp = vdir / f"upload.{ext}"

    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    stream_cap = settings.STREAM_CAP_MB * 1024 * 1024
    size = 0
    try:
        with open(tmp, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > stream_cap:
                    raise HTTPException(
                        413,
                        f"File exceeds the {settings.STREAM_CAP_MB} MB safety cap. "
                        "Please trim it before uploading.",
                    )
                out.write(chunk)
    except HTTPException:
        tmp.unlink(missing_ok=True)
        raise
    finally:
        await file.close()

    trimmed = False
    original_mb = round(size / (1024 * 1024), 1)
    if size > max_bytes:
        # Keep only the first part of the video, up to TRIM_MB megabytes.
        try:
            trim_to_size(tmp, dest, settings.TRIM_MB)
            trimmed = True
        except RuntimeError as e:
            tmp.unlink(missing_ok=True)
            raise HTTPException(422, f"Could not trim oversized upload: {e}")
        tmp.unlink(missing_ok=True)
    else:
        tmp.rename(dest)

    final_mb = round(dest.stat().st_size / (1024 * 1024), 1)
    ingest = {
        "video_id": video_id,
        "filename": file.filename,
        "duration": _probe_duration(dest),
        "status": "uploaded",
        "trimmed": trimmed,
        "original_size_mb": original_mb,
        "size_mb": final_mb,
    }
    stage_cache(video_id).save("ingest", ingest)
    set_progress(video_id, "idle", 0, 0)
    response: dict = {"video_id": video_id, "filename": file.filename, "trimmed": trimmed}
    if trimmed:
        response["note"] = (
            f"Your video was {original_mb} MB, over the {settings.MAX_UPLOAD_MB} MB limit, "
            f"so only the first {final_mb} MB will be processed."
        )
    return response


@router.get("/videos/{video_id}")
async def get_video(video_id: str) -> dict:
    """Return ingest info plus which pipeline stage caches exist."""
    _require_video(video_id)
    cache = stage_cache(video_id)
    stages = {name: cache.exists(name) for name in STAGE_NAMES}
    clips = cache.load("clips") or []
    return {
        "ingest": cache.load("ingest"),
        "stages": stages,
        "clip_count": len(clips),
        "progress": read_progress(video_id),
    }


@router.post("/videos/{video_id}/process")
async def process_video(
    video_id: str,
    background: BackgroundTasks,
    n: int = Query(8, ge=1, le=30),
    fresh: bool = Query(False),
) -> dict:
    """Launch the pipeline in the background. Returns immediately."""
    _require_video(video_id)
    set_progress(video_id, "queued", 0, 0)
    background.add_task(run_pipeline, video_id, n, fresh)
    return {"job": "started"}


@router.get("/videos/{video_id}/progress")
async def get_progress(video_id: str) -> dict:
    """Poll pipeline progress: stage, done, total, and error if any."""
    _require_video(video_id)
    return read_progress(video_id)


@router.get("/videos/{video_id}/clips")
async def list_clips(video_id: str) -> list[dict]:
    """List all clip records for a video."""
    _require_video(video_id)
    clips = stage_cache(video_id).load("clips") or []
    return [ClipRecord.model_validate(c).model_dump() for c in clips]


def _load_clip_records(video_id: str) -> list[ClipRecord]:
    clips = stage_cache(video_id).load("clips") or []
    return [ClipRecord.model_validate(c) for c in clips]


def _save_clip_records(video_id: str, clips: list[ClipRecord]) -> None:
    stage_cache(video_id).save("clips", [c.model_dump() for c in clips])


@router.post("/videos/{video_id}/clips/{clip_id}/rerender")
async def rerender_clip(
    video_id: str, clip_id: str, body: RerenderRequest
) -> dict:
    """Rebuild one clip with adjusted start, end, or title, then re-render."""
    _require_video(video_id)
    clips = _load_clip_records(video_id)
    clip = next((c for c in clips if c.id == clip_id), None)
    if clip is None:
        raise HTTPException(404, f"Clip {clip_id} not found")

    cand = clip.candidate
    if body.start is not None:
        cand.start = max(0.0, body.start)
    if body.end is not None:
        cand.end = max(cand.start + 1.0, body.end)
    if body.title is not None:
        cand.title = body.title[:80]
    clip.candidate = Candidate.model_validate(cand.model_dump())
    clip.status = "pending"
    clip.error = None

    render_clip(video_id, clip, work_dir())
    _save_clip_records(video_id, clips)
    return clip.model_dump()


@router.get("/videos/{video_id}/download/{clip_id}")
async def download_clip(video_id: str, clip_id: str) -> FileResponse:
    """Download the rendered mp4 for one clip."""
    _require_video(video_id)
    clips = _load_clip_records(video_id)
    clip = next((c for c in clips if c.id == clip_id), None)
    if clip is None:
        raise HTTPException(404, f"Clip {clip_id} not found")
    if not clip.output_path or not Path(clip.output_path).exists():
        raise HTTPException(
            409,
            f"Clip {clip_id} has no rendered file yet (status: {clip.status}).",
        )
    return FileResponse(
        clip.output_path,
        media_type="video/mp4",
        filename=f"{clip_id}.mp4",
    )


@router.get("/videos/{video_id}/captions/{clip_id}")
async def download_captions(video_id: str, clip_id: str) -> FileResponse:
    """Download the SRT subtitle file for one clip."""
    _require_video(video_id)
    clips = _load_clip_records(video_id)
    clip = next((c for c in clips if c.id == clip_id), None)
    if clip is None:
        raise HTTPException(404, f"Clip {clip_id} not found")
    if not clip.srt_path or not Path(clip.srt_path).exists():
        raise HTTPException(
            409,
            f"Clip {clip_id} has no SRT file yet (status: {clip.status}).",
        )
    return FileResponse(
        clip.srt_path,
        media_type="text/plain",
        filename=f"{clip_id}.srt",
    )
