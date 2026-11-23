"""One-time historical data backfill script."""

import asyncio
import logging

from puntboom.collectors.football import FootballCollector, COMPETITIONS
from puntboom.collectors.basketball import BasketballCollector
from puntboom.collectors.hockey import HockeyCollector
from puntboom.collectors.tennis import TennisCollector
from puntboom.database import async_session, init_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


async def seed_football():
    logger.info("Seeding football data...")
    collector = FootballCollector()
    async with async_session() as session:
        await collector.collect_competitions(session)
        for code in COMPETITIONS:
            try:
                await collector.collect_matches(session, code)
                logger.info(f"  Seeded {code}")
            except Exception as e:
                logger.error(f"  Failed {code}: {e}")
    await collector.close()


async def seed_basketball():
    logger.info("Seeding basketball data...")
    collector = BasketballCollector()
    async with async_session() as session:
        await collector.collect_teams(session)
        for season in [2023, 2024]:
            try:
                await collector.collect_games(session, season=season)
                logger.info(f"  Seeded season {season}")
            except Exception as e:
                logger.error(f"  Failed season {season}: {e}")
    await collector.close()


async def seed_hockey():
    logger.info("Seeding hockey data...")
    collector = HockeyCollector()
    async with async_session() as session:
        await collector.collect_teams(session)
        await collector.collect_schedule(session)
        logger.info("  Seeded NHL schedule")
    await collector.close()


async def seed_tennis():
    logger.info("Seeding tennis data...")
    collector = TennisCollector()
    async with async_session() as session:
        await collector.collect_scoreboard(session, "atp")
        await collector.collect_scoreboard(session, "wta")
        logger.info("  Seeded ATP/WTA scoreboards")
    await collector.close()


async def main():
    logger.info("Initializing database...")
    await init_db()

    await seed_football()
    await seed_basketball()
    await seed_hockey()
    await seed_tennis()

    logger.info("Historical data seeding complete!")


if __name__ == "__main__":
    asyncio.run(main())
