from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.config import settings
from backend.core.model_loader import get_yolo_model
from backend.api.routes import (
    detection_router,
    interactive_router,
    nlp_router,
    stream_router,
    system_router,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm model on startup so the first request is instant
    print("[STARTUP] Pre-warming YOLO11m model...")
    get_yolo_model()
    print("[STARTUP] Backend ready for requests!")
    yield
    print("[SHUTDOWN] Backend shutting down.")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="YOLO11m Local Detection Backend",
        description=(
            "Fully local, Apple Silicon MPS-accelerated object detection API "
            "for images, videos, and real-time webcam streams."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # Enable CORS for all origins (React, Vue, Vite, Next.js, etc.)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount static directories
    app.mount("/outputs", StaticFiles(directory=str(settings.outputs_dir)), name="outputs")
    app.mount("/static", StaticFiles(directory=str(settings.static_dir)), name="static")

    # Include API routers
    app.include_router(system_router, prefix="/api")
    app.include_router(detection_router, prefix="/api")
    app.include_router(stream_router, prefix="/api")
    app.include_router(nlp_router, prefix="/api")
    app.include_router(interactive_router, prefix="/api")

    @app.get("/", include_in_schema=False)
    def index():
        """Serve the built-in test web UI dashboard."""
        index_file = settings.static_dir / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return {
            "message": "YOLO11m Local Detection Backend is running.",
            "docs": "/docs",
        }

    return app


app = create_app()
