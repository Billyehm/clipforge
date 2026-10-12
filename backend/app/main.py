"""ClipForge FastAPI application entry point."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.config import work_dir


def create_app() -> FastAPI:
    """Build the FastAPI app with CORS, static files, and API routes."""
    app = FastAPI(title="ClipForge API", version="0.2.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    work = work_dir()
    work.mkdir(parents=True, exist_ok=True)
    app.mount("/files", StaticFiles(directory=str(work)), name="files")

    app.include_router(router, prefix="/api")

    @app.get("/health")
    def health() -> dict:
        return {"ok": True}

    return app


app = create_app()
