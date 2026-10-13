"""Application configuration loaded from environment variables."""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the clipforge backend."""

    GROQ_API_KEY: str = ""
    WORK_DIR: str = "work"
    MAX_UPLOAD_MB: int = 500
    # Oversized uploads are trimmed to this size (first part of the video).
    TRIM_MB: int = 499
    # Hard cap on the incoming upload stream, as a safety net so one giant
    # upload cannot fill the disk before trimming happens.
    STREAM_CAP_MB: int = 2048
    # How many transcription chunks are sent to Groq in parallel.
    TRANSCRIBE_WORKERS: int = 3

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()


def work_dir() -> Path:
    """Resolve the working directory.

    A relative WORK_DIR is anchored at the repo root (the parent of the
    backend/ directory), so uploads and outputs land in clipforge/work/
    no matter where the server is started from.
    """
    p = Path(settings.WORK_DIR)
    if not p.is_absolute():
        repo_root = Path(__file__).resolve().parents[3]
        p = repo_root / p
    return p
