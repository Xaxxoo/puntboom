"""FastAPI application factory and entry point."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from puntboom.api.router import api_router
from puntboom.config import settings
from puntboom.database import init_db
from puntboom.web.views import router as web_router

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: init DB and start scheduler."""
    logger.info("Starting Puntboom...")
    await init_db()
    settings.model_dir.mkdir(parents=True, exist_ok=True)

    # Start scheduler
    try:
        from puntboom.tasks.scheduler import setup_scheduler

        scheduler = setup_scheduler()
        scheduler.start()
        logger.info("Scheduler started")
    except Exception:
        logger.exception("Failed to start scheduler")

    yield

    # Shutdown
    try:
        scheduler.shutdown(wait=False)
    except Exception:
        pass
    logger.info("Puntboom stopped")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Puntboom",
        description="Sports Betting Prediction Platform",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Mount static files - resolve to project root (puntboom/)
    project_root = Path(__file__).resolve().parent.parent.parent
    static_dir = project_root / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Include routers
    app.include_router(api_router)
    app.include_router(web_router)

    return app


app = create_app()


def run():
    """Entry point for the puntboom CLI command."""
    uvicorn.run(
        "puntboom.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )


if __name__ == "__main__":
    run()
