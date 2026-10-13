"""Small media helpers: trimming, probing."""
import subprocess
from pathlib import Path


def trim_to_size(src: Path, dest: Path, max_mb: int) -> None:
    """Keep only the first part of a video, up to max_mb megabytes.

    Uses stream copy with ffmpeg's -fs flag, so the output is a valid,
    playable file containing the beginning of the source. Raises
    RuntimeError if ffmpeg fails.
    """
    proc = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(src),
            "-c", "copy",
            "-fs", f"{max_mb}M",
            str(dest),
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if proc.returncode != 0 or not dest.exists() or dest.stat().st_size == 0:
        tail = (proc.stderr or "")[-2000:]
        raise RuntimeError(f"ffmpeg trim failed: {tail}")
