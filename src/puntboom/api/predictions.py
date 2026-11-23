"""Predictions API endpoints."""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select, and_, func
from sqlalchemy.ext.asyncio import AsyncSession

from puntboom.database import get_session
from puntboom.models.prediction import Prediction
from puntboom.schemas.prediction import PredictionListResponse, PredictionSchema

router = APIRouter(prefix="/api/predictions", tags=["predictions"])


@router.get("/{sport}", response_model=PredictionListResponse)
async def get_predictions(
    sport: str,
    upcoming_only: bool = Query(False, description="Only show upcoming matches"),
    value_only: bool = Query(False, description="Only show value bets"),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
    limit: int = Query(50, ge=1, le=200),
    session: AsyncSession = Depends(get_session),
):
    """Get predictions for a sport."""
    query = select(Prediction).where(Prediction.sport == sport)

    if upcoming_only:
        query = query.where(
            and_(
                Prediction.actual_outcome.is_(None),
                Prediction.match_date >= datetime.now(timezone.utc),
            )
        )
    if value_only:
        query = query.where(Prediction.is_value_bet == True)
    if min_confidence > 0:
        query = query.where(Prediction.confidence >= min_confidence)

    query = query.order_by(Prediction.match_date.desc()).limit(limit)
    result = await session.execute(query)
    predictions = result.scalars().all()

    # Count total
    count_query = select(func.count(Prediction.id)).where(Prediction.sport == sport)
    count_result = await session.execute(count_query)
    total = count_result.scalar() or 0

    return PredictionListResponse(
        predictions=[PredictionSchema.model_validate(p) for p in predictions],
        total=total,
        sport=sport,
    )


@router.get("/", response_model=PredictionListResponse)
async def get_all_predictions(
    value_only: bool = Query(False),
    limit: int = Query(50, ge=1, le=200),
    session: AsyncSession = Depends(get_session),
):
    """Get predictions across all sports."""
    query = select(Prediction)
    if value_only:
        query = query.where(Prediction.is_value_bet == True)
    query = query.order_by(Prediction.match_date.desc()).limit(limit)

    result = await session.execute(query)
    predictions = result.scalars().all()

    count_result = await session.execute(select(func.count(Prediction.id)))
    total = count_result.scalar() or 0

    return PredictionListResponse(
        predictions=[PredictionSchema.model_validate(p) for p in predictions],
        total=total,
    )
