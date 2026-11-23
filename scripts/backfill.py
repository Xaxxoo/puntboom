"""Historical data backfill script.

Pulls historical data through the existing collector classes so ML models
can train immediately (~2000+ finished matches per sport).

Usage:
    python scripts/backfill.py                        # all sports
    python scripts/backfill.py --sport football       # single sport
    python scripts/backfill.py --sport hockey tennis  # multiple
    python scripts/backfill.py --retrain              # backfill + retrain models
"""

import argparse
import asyncio
import logging
import sys
import time
from datetime import date, timedelta

# Allow running from repo root: `python scripts/backfill.py`
sys.path.insert(0, "src")

from puntboom.collectors.football import COMPETITIONS, FootballCollector
from puntboom.collectors.basketball import BasketballCollector
from puntboom.collectors.hockey import HockeyCollector
from puntboom.collectors.tennis import TennisCollector
from puntboom.database import async_session, init_db

logger = logging.getLogger("backfill")

ALL_SPORTS = ["football", "basketball", "hockey", "tennis"]

# ---------------------------------------------------------------------------
# Football: 3 seasons × 6 competitions
# ---------------------------------------------------------------------------

async def backfill_football() -> int:
    """Backfill football matches for seasons 2022-2024."""
    seasons = [2022, 2023, 2024]
    total = 0
    collector = FootballCollector()
    try:
        async with async_session() as session:
            logger.info("Football: collecting competitions …")
            await collector.collect_competitions(session)

            for season in seasons:
                for code in COMPETITIONS:
                    try:
                        matches = await collector.collect_matches(
                            session, competition=code, season=season
                        )
                        total += len(matches)
                        logger.info(
                            f"  {code} {season}: {len(matches)} matches"
                        )
                    except Exception:
                        logger.exception(
                            f"  {code} {season}: FAILED — continuing"
                        )
    finally:
        await collector.close()
    return total


# ---------------------------------------------------------------------------
# Basketball: 2 seasons (pagination handled internally)
# ---------------------------------------------------------------------------

async def backfill_basketball() -> int:
    """Backfill basketball games for seasons 2023-2024."""
    seasons = [2023, 2024]
    total = 0
    collector = BasketballCollector()
    try:
        async with async_session() as session:
            logger.info("Basketball: collecting teams …")
            await collector.collect_teams(session)

            for season in seasons:
                try:
                    games = await collector.collect_games(session, season=season)
                    total += len(games)
                    logger.info(f"  Season {season}: {len(games)} games")
                except Exception:
                    logger.exception(
                        f"  Season {season}: FAILED — continuing"
                    )
    finally:
        await collector.close()
    return total


# ---------------------------------------------------------------------------
# Hockey: step by 7 days from Oct 2024 → today
# ---------------------------------------------------------------------------

async def backfill_hockey() -> int:
    """Backfill hockey schedule from Oct 2024 to today."""
    start = date(2024, 10, 1)
    end = date.today()
    total = 0
    collector = HockeyCollector()
    try:
        async with async_session() as session:
            logger.info("Hockey: collecting teams …")
            await collector.collect_teams(session)

            current = start
            while current <= end:
                date_str = current.isoformat()  # YYYY-MM-DD
                try:
                    games = await collector.collect_schedule(
                        session, date=date_str
                    )
                    total += len(games)
                    logger.info(f"  Week of {date_str}: {len(games)} games")
                except Exception:
                    logger.exception(
                        f"  Week of {date_str}: FAILED — continuing"
                    )
                current += timedelta(days=7)
    finally:
        await collector.close()
    return total


# ---------------------------------------------------------------------------
# Tennis: daily calls for ATP + WTA from Jan 2025 → today
# ---------------------------------------------------------------------------

async def backfill_tennis() -> int:
    """Backfill tennis scoreboards (ATP + WTA) from Jan 2025 to today."""
    start = date(2025, 1, 1)
    end = date.today()
    tours = ["atp", "wta"]
    total = 0
    days_done = 0
    collector = TennisCollector()
    try:
        async with async_session() as session:
            current = start
            while current <= end:
                date_str = current.strftime("%Y%m%d")  # YYYYMMDD
                for tour in tours:
                    try:
                        matches = await collector.collect_scoreboard(
                            session, tour=tour, date=date_str
                        )
                        total += len(matches)
                    except Exception:
                        logger.exception(
                            f"  {tour.upper()} {current.isoformat()}: FAILED — continuing"
                        )
                days_done += 1
                # Log progress every 30 days
                if days_done % 30 == 0:
                    logger.info(
                        f"  Tennis progress: {days_done} days processed, "
                        f"{total} matches so far"
                    )
                current += timedelta(days=1)
    finally:
        await collector.close()
    return total


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

SPORT_FUNCS = {
    "football": backfill_football,
    "basketball": backfill_basketball,
    "hockey": backfill_hockey,
    "tennis": backfill_tennis,
}


async def main(sports: list[str], retrain: bool) -> None:
    await init_db()

    results: dict[str, dict] = {}

    for sport in sports:
        func = SPORT_FUNCS[sport]
        logger.info(f"\n{'='*50}")
        logger.info(f"Starting backfill: {sport.upper()}")
        logger.info(f"{'='*50}")
        t0 = time.monotonic()
        try:
            count = await func()
            elapsed = time.monotonic() - t0
            results[sport] = {"count": count, "elapsed": elapsed, "status": "OK"}
        except Exception:
            elapsed = time.monotonic() - t0
            logger.exception(f"{sport} backfill failed")
            results[sport] = {"count": 0, "elapsed": elapsed, "status": "ERROR"}

    # Print summary table
    print(f"\n{'='*55}")
    print(f" {'Sport':<14}{'Records':>10}{'Time':>12}  {'Status'}")
    print(f"{'-'*55}")
    total_records = 0
    total_time = 0.0
    for sport, info in results.items():
        mins = info["elapsed"] / 60
        total_records += info["count"]
        total_time += info["elapsed"]
        print(
            f" {sport:<14}{info['count']:>10}{mins:>10.1f}m  {info['status']}"
        )
    print(f"{'-'*55}")
    print(
        f" {'TOTAL':<14}{total_records:>10}{total_time / 60:>10.1f}m"
    )
    print(f"{'='*55}\n")

    if retrain:
        logger.info("Retraining models …")
        from puntboom.tasks.retrain import retrain_all_models

        retrain_results = await retrain_all_models()
        for sport, result in retrain_results.items():
            logger.info(f"  {sport}: {result}")
        logger.info("Retraining complete.")


def cli() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill historical sports data for ML training."
    )
    parser.add_argument(
        "--sport",
        nargs="+",
        choices=ALL_SPORTS,
        default=ALL_SPORTS,
        help="Sports to backfill (default: all).",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Retrain ML models after backfill.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    asyncio.run(main(args.sport, args.retrain))


if __name__ == "__main__":
    cli()
