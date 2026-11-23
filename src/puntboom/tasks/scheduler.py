"""APScheduler setup for background tasks."""

import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from puntboom.config import settings

logger = logging.getLogger(__name__)

scheduler = AsyncIOScheduler()


def setup_scheduler() -> AsyncIOScheduler:
    """Configure and return the scheduler with all jobs."""
    from puntboom.tasks.collect import collect_all_data
    from puntboom.tasks.predict import generate_predictions
    from puntboom.tasks.retrain import retrain_all_models

    # Collect data every 2 hours
    scheduler.add_job(
        collect_all_data,
        trigger=IntervalTrigger(hours=2),
        id="collect_data",
        name="Collect sports data",
        replace_existing=True,
    )

    # Generate predictions every 3 hours
    scheduler.add_job(
        generate_predictions,
        trigger=IntervalTrigger(hours=3),
        id="generate_predictions",
        name="Generate predictions",
        replace_existing=True,
    )

    # Retrain models daily
    scheduler.add_job(
        retrain_all_models,
        trigger=IntervalTrigger(hours=settings.retrain_interval_hours),
        id="retrain_models",
        name="Retrain ML models",
        replace_existing=True,
    )

    logger.info("Scheduler configured with all jobs")
    return scheduler
