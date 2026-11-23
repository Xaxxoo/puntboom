"""Generate predictions for upcoming matches."""

import asyncio
import logging

from puntboom.database import init_db
from puntboom.tasks.predict import generate_predictions

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


async def main():
    logger.info("Initializing database...")
    await init_db()

    logger.info("Generating predictions...")
    await generate_predictions()
    logger.info("Prediction generation complete!")


if __name__ == "__main__":
    asyncio.run(main())
