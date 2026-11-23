"""Bankroll P&L tracking over time."""

from datetime import datetime, timezone

from sqlalchemy import select, and_, func
from sqlalchemy.ext.asyncio import AsyncSession

from puntboom.models.prediction import Prediction


async def get_cumulative_pnl(
    session: AsyncSession,
    sport: str | None = None,
    bankroll: float = 1000.0,
) -> list[dict]:
    """Calculate cumulative P&L from resolved predictions with kelly stakes.

    Returns list of dicts with date, cumulative_pnl, bankroll_value.
    """
    query = select(Prediction).where(
        and_(
            Prediction.actual_outcome.isnot(None),
            Prediction.kelly_stake.isnot(None),
            Prediction.kelly_stake > 0,
        )
    )
    if sport:
        query = query.where(Prediction.sport == sport)
    query = query.order_by(Prediction.match_date)

    result = await session.execute(query)
    predictions = result.scalars().all()

    cumulative_pnl = 0.0
    history = []

    for pred in predictions:
        stake = pred.kelly_stake or 0.0
        if pred.is_correct:
            # Find which odds to use based on predicted outcome
            if pred.predicted_outcome in ("H", "1"):
                odds = pred.odds_home or 0.0
            elif pred.predicted_outcome == "D":
                odds = pred.odds_draw or 0.0
            else:
                odds = pred.odds_away or 0.0
            profit = stake * (odds - 1)
        else:
            profit = -stake

        cumulative_pnl += profit
        history.append(
            {
                "date": pred.match_date.isoformat() if pred.match_date else "",
                "sport": pred.sport,
                "match": f"{pred.home_team} vs {pred.away_team}",
                "predicted": pred.predicted_outcome,
                "actual": pred.actual_outcome,
                "stake": stake,
                "profit": profit,
                "cumulative_pnl": cumulative_pnl,
                "bankroll_value": bankroll + cumulative_pnl,
            }
        )

    return history


async def get_summary_stats(
    session: AsyncSession, sport: str | None = None
) -> dict:
    """Get summary P&L statistics."""
    query = select(Prediction).where(
        and_(
            Prediction.actual_outcome.isnot(None),
            Prediction.is_value_bet == True,
        )
    )
    if sport:
        query = query.where(Prediction.sport == sport)

    result = await session.execute(query)
    predictions = result.scalars().all()

    total = len(predictions)
    correct = sum(1 for p in predictions if p.is_correct)
    total_staked = sum(p.kelly_stake or 0 for p in predictions if p.kelly_stake)

    total_return = 0.0
    for p in predictions:
        if not p.kelly_stake:
            continue
        if p.is_correct:
            if p.predicted_outcome in ("H", "1"):
                odds = p.odds_home or 0.0
            elif p.predicted_outcome == "D":
                odds = p.odds_draw or 0.0
            else:
                odds = p.odds_away or 0.0
            total_return += p.kelly_stake * odds
        # Losses: return is 0 (stake already counted)

    return {
        "total_predictions": total,
        "correct": correct,
        "accuracy": correct / max(total, 1),
        "total_staked": total_staked,
        "total_return": total_return,
        "profit": total_return - total_staked,
        "roi": (total_return - total_staked) / max(total_staked, 1),
    }
