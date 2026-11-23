"""Bankroll API endpoints."""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from puntboom.bankroll.kelly import kelly_stake_amount, fractional_kelly
from puntboom.bankroll.tracker import get_cumulative_pnl, get_summary_stats
from puntboom.config import settings
from puntboom.database import get_session
from puntboom.models.prediction import Prediction
from puntboom.schemas.bankroll import BankrollSummary, BetRecommendation, PnLEntry

router = APIRouter(prefix="/api/bankroll", tags=["bankroll"])


@router.get("/recommendations")
async def get_recommendations(
    bankroll: float = Query(default=None),
    sport: str | None = Query(default=None),
    session: AsyncSession = Depends(get_session),
):
    """Get current bet recommendations based on value bets."""
    if bankroll is None:
        bankroll = settings.default_bankroll

    query = select(Prediction).where(
        and_(
            Prediction.is_value_bet == True,
            Prediction.actual_outcome.is_(None),
            Prediction.match_date >= datetime.now(timezone.utc),
        )
    )
    if sport:
        query = query.where(Prediction.sport == sport)
    query = query.order_by(Prediction.expected_value.desc()).limit(20)

    result = await session.execute(query)
    predictions = result.scalars().all()

    recommendations = []
    for pred in predictions:
        if pred.predicted_outcome in ("H", "1"):
            odds = pred.odds_home or 0.0
        elif pred.predicted_outcome == "D":
            odds = pred.odds_draw or 0.0
        else:
            odds = pred.odds_away or 0.0

        if odds <= 1.0:
            continue

        stake = kelly_stake_amount(pred.confidence, odds, bankroll)
        if stake <= 0:
            continue

        recommendations.append(
            BetRecommendation(
                sport=pred.sport,
                match=f"{pred.home_team} vs {pred.away_team}",
                match_date=pred.match_date.isoformat() if pred.match_date else "",
                outcome=pred.predicted_outcome,
                probability=pred.confidence,
                odds=odds,
                expected_value=pred.expected_value or 0.0,
                kelly_fraction=fractional_kelly(pred.confidence, odds),
                stake_amount=round(stake, 2),
            )
        )

    return {"bankroll": bankroll, "recommendations": recommendations}


@router.get("/summary")
async def get_bankroll_summary(
    sport: str | None = Query(default=None),
    session: AsyncSession = Depends(get_session),
):
    """Get bankroll summary statistics."""
    stats = await get_summary_stats(session, sport)
    return BankrollSummary(
        bankroll=settings.default_bankroll,
        **stats,
    )


@router.get("/pnl")
async def get_pnl_history(
    sport: str | None = Query(default=None),
    session: AsyncSession = Depends(get_session),
):
    """Get cumulative P&L history."""
    history = await get_cumulative_pnl(
        session, sport, bankroll=settings.default_bankroll
    )
    return {
        "history": [PnLEntry(**entry) for entry in history],
        "starting_bankroll": settings.default_bankroll,
    }
