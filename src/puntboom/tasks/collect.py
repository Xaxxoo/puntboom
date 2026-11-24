"""Scheduled data collection jobs."""

import logging

from puntboom.collectors.football import FootballCollector
from puntboom.collectors.basketball import BasketballCollector
from puntboom.collectors.hockey import HockeyCollector
from puntboom.collectors.euro_hockey import EuroHockeyCollector
from puntboom.collectors.tennis import TennisCollector
from puntboom.collectors.odds import OddsCollector
from puntboom.database import async_session

logger = logging.getLogger(__name__)


async def collect_all_data() -> None:
    """Run all data collectors."""
    logger.info("Starting scheduled data collection")

    async with async_session() as session:
        collectors = [
            ("football", FootballCollector()),
            ("basketball", BasketballCollector()),
            ("hockey", HockeyCollector()),
            ("euro_hockey", EuroHockeyCollector()),
            ("tennis", TennisCollector()),
            ("odds", OddsCollector()),
        ]
        for name, collector in collectors:
            try:
                await collector.collect_all(session)
                logger.info(f"Completed {name} collection")
            except Exception:
                logger.exception(f"Failed {name} collection")
            finally:
                await collector.close()

    logger.info("Finished scheduled data collection")


async def collect_sport(sport: str) -> None:
    """Collect data for a single sport."""
    collector_map = {
        "football": FootballCollector,
        "basketball": BasketballCollector,
        "hockey": HockeyCollector,
        "euro_hockey": EuroHockeyCollector,
        "tennis": TennisCollector,
        "odds": OddsCollector,
    }
    cls = collector_map.get(sport)
    if not cls:
        logger.error(f"Unknown sport: {sport}")
        return

    async with async_session() as session:
        collector = cls()
        try:
            await collector.collect_all(session)
        finally:
            await collector.close()
