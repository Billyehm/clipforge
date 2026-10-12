"""Per-video JSON cache so each pipeline stage can be re-run on its own.

Layout: work/<video_id>/<stage>.json
Writes are atomic: data goes to a temp file first, then os.replace
swaps it in, so a crash never leaves a half-written stage file.
"""

import json
import os
import tempfile
from pathlib import Path

STAGES = ("ingest", "transcript", "candidates", "clips")


def _default_root() -> Path:
    # backend/app/pipeline -> repo root is four levels up
    return Path(__file__).resolve().parent.parent.parent.parent / "work"


class StageCache:
    """JSON stage cache rooted at work/<video_id>/."""

    def __init__(self, video_id: str, root: str | Path | None = None) -> None:
        self.video_id = video_id
        base = Path(root) if root is not None else _default_root()
        self.dir = base / video_id
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, stage_name: str) -> Path:
        if stage_name not in STAGES:
            raise ValueError(f"Unknown stage {stage_name!r}. Use one of {STAGES}.")
        return self.dir / f"{stage_name}.json"

    def save(self, stage_name: str, data: dict | list) -> Path:
        """Write stage data atomically. Returns the final file path."""
        path = self._path(stage_name)
        fd, tmp = tempfile.mkstemp(dir=self.dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return path

    def load(self, stage_name: str) -> dict | list | None:
        """Return the cached data, or None when the stage has not run yet."""
        path = self._path(stage_name)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def exists(self, stage_name: str) -> bool:
        return self._path(stage_name).exists()
