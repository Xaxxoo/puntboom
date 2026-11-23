"""Train all sport models."""

import asyncio
import json
import logging

from puntboom.database import init_db
from puntboom.tasks.retrain import retrain_all_models

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


async def main():
    logger.info("Initializing database...")
    await init_db()

    logger.info("Training all models...")
    results = await retrain_all_models()

    for sport, result in results.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"  {sport.upper()}")
        logger.info(f"{'='*50}")
        if "error" in result:
            logger.warning(f"  Error: {result['error']}")
        else:
            logger.info(f"  Best model: {result.get('best_model', 'N/A')}")
            logger.info(f"  Samples: {result.get('samples', 0)}")
            metrics = result.get("metrics", {})
            if "mae" in metrics:
                logger.info(f"  MAE: {metrics['mae']:.4f}")
                logger.info(f"  Avg total goals: {metrics.get('avg_total_goals', 0):.2f}")
            else:
                logger.info(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
                logger.info(f"  Brier score: {metrics.get('brier_score', 0):.4f}")
                logger.info(f"  Log loss: {metrics.get('log_loss', 0):.4f}")

    logger.info("\nTraining complete!")


if __name__ == "__main__":
    asyncio.run(main())
