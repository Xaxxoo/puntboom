"""Model performance API endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from puntboom.database import get_session
from puntboom.models.prediction import ModelPerformance, Prediction
from puntboom.schemas.prediction import ModelPerformanceSchema

router = APIRouter(prefix="/api/performance", tags=["performance"])


@router.get("/{sport}")
async def get_performance(
    sport: str,
    session: AsyncSession = Depends(get_session),
):
    """Get model performance metrics for a sport."""
    result = await session.execute(
        select(ModelPerformance)
        .where(ModelPerformance.sport == sport)
        .order_by(ModelPerformance.trained_at.desc())
        .limit(10)
    )
    entries = result.scalars().all()

    # Also compute live performance from predictions
    pred_result = await session.execute(
        select(Prediction).where(
            Prediction.sport == sport,
            Prediction.actual_outcome.isnot(None),
        )
    )
    predictions = pred_result.scalars().all()

    total = len(predictions)
    correct = sum(1 for p in predictions if p.is_correct)
    value_bets = [p for p in predictions if p.is_value_bet]
    value_correct = sum(1 for p in value_bets if p.is_correct)

    return {
        "sport": sport,
        "model_history": [
            ModelPerformanceSchema.model_validate(e) for e in entries
        ],
        "live_stats": {
            "total_predictions": total,
            "correct": correct,
            "accuracy": correct / max(total, 1),
            "value_bets": len(value_bets),
            "value_correct": value_correct,
            "value_accuracy": value_correct / max(len(value_bets), 1),
        },
    }


@router.get("/")
async def get_all_performance(
    session: AsyncSession = Depends(get_session),
):
    """Get performance summary across all sports."""
    sports = ["football", "basketball", "hockey", "tennis"]
    summary = {}

    for sport in sports:
        pred_result = await session.execute(
            select(Prediction).where(
                Prediction.sport == sport,
                Prediction.actual_outcome.isnot(None),
            )
        )
        predictions = pred_result.scalars().all()
        total = len(predictions)
        correct = sum(1 for p in predictions if p.is_correct)
        summary[sport] = {
            "total": total,
            "correct": correct,
            "accuracy": correct / max(total, 1),
        }

    return {"performance": summary}
