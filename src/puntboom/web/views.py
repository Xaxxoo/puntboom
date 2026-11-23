"""HTML page routes (server-rendered with Jinja2)."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, Request
from fastapi.templating import Jinja2Templates
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from puntboom.database import get_session
from puntboom.models.prediction import Prediction

router = APIRouter(tags=["web"])
_project_root = Path(__file__).resolve().parent.parent.parent.parent
templates = Jinja2Templates(directory=str(_project_root / "templates"))

SPORTS = [
    {"key": "football", "name": "Football", "icon": "\u26bd"},
    {"key": "basketball", "name": "Basketball", "icon": "\U0001f3c0"},
    {"key": "hockey", "name": "Ice Hockey", "icon": "\U0001f3d2"},
    {"key": "tennis", "name": "Tennis", "icon": "\U0001f3be"},
]


@router.get("/")
async def dashboard(
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Main dashboard page."""
    sport_stats = {}
    for sport in SPORTS:
        key = sport["key"]
        total_res = await session.execute(
            select(func.count(Prediction.id)).where(
                Prediction.sport == key, Prediction.actual_outcome.isnot(None)
            )
        )
        correct_res = await session.execute(
            select(func.count(Prediction.id)).where(
                Prediction.sport == key, Prediction.is_correct == True
            )
        )
        total = total_res.scalar() or 0
        correct = correct_res.scalar() or 0

        # O/U stats
        ou_total_res = await session.execute(
            select(func.count(Prediction.id)).where(
                Prediction.sport == key, Prediction.actual_total.isnot(None)
            )
        )
        ou_correct_res = await session.execute(
            select(func.count(Prediction.id)).where(
                Prediction.sport == key, Prediction.is_total_correct == True
            )
        )
        ou_total = ou_total_res.scalar() or 0
        ou_correct = ou_correct_res.scalar() or 0

        sport_stats[key] = {
            "total": total,
            "correct": correct,
            "accuracy": f"{correct / max(total, 1) * 100:.1f}%",
            "ou_total": ou_total,
            "ou_correct": ou_correct,
            "ou_accuracy": f"{ou_correct / max(ou_total, 1) * 100:.1f}%",
        }

    # Today's value bets
    now = datetime.now(timezone.utc)
    value_res = await session.execute(
        select(Prediction)
        .where(
            and_(
                Prediction.is_value_bet == True,
                Prediction.actual_outcome.is_(None),
            )
        )
        .order_by(Prediction.expected_value.desc())
        .limit(10)
    )
    value_bets = value_res.scalars().all()

    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "sports": SPORTS,
            "sport_stats": sport_stats,
            "value_bets": value_bets,
        },
    )


@router.get("/sport/{sport_key}")
async def sport_page(
    request: Request,
    sport_key: str,
    session: AsyncSession = Depends(get_session),
):
    """Sport-specific page."""
    sport = next((s for s in SPORTS if s["key"] == sport_key), None)
    if not sport:
        sport = {"key": sport_key, "name": sport_key.title(), "icon": ""}

    # Upcoming predictions
    upcoming_res = await session.execute(
        select(Prediction)
        .where(
            and_(
                Prediction.sport == sport_key,
                Prediction.actual_outcome.is_(None),
            )
        )
        .order_by(Prediction.match_date)
        .limit(20)
    )
    upcoming = upcoming_res.scalars().all()

    # Recent results
    recent_res = await session.execute(
        select(Prediction)
        .where(
            and_(
                Prediction.sport == sport_key,
                Prediction.actual_outcome.isnot(None),
            )
        )
        .order_by(Prediction.match_date.desc())
        .limit(20)
    )
    recent = recent_res.scalars().all()

    return templates.TemplateResponse(
        request,
        "sport.html",
        {
            "sport": sport,
            "upcoming": upcoming,
            "recent": recent,
            "sports": SPORTS,
        },
    )


@router.get("/predictions")
async def predictions_page(
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Full predictions page with filters."""
    preds_res = await session.execute(
        select(Prediction)
        .order_by(Prediction.match_date.desc())
        .limit(100)
    )
    predictions = preds_res.scalars().all()

    return templates.TemplateResponse(
        request,
        "predictions.html",
        {
            "predictions": predictions,
            "sports": SPORTS,
        },
    )


@router.get("/performance")
async def performance_page(
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Model performance page."""
    sport_perf = {}
    for sport in SPORTS:
        key = sport["key"]
        res = await session.execute(
            select(Prediction).where(
                Prediction.sport == key,
                Prediction.actual_outcome.isnot(None),
            )
        )
        preds = res.scalars().all()
        total = len(preds)
        correct = sum(1 for p in preds if p.is_correct)
        ou_total = sum(1 for p in preds if p.actual_total is not None)
        ou_correct = sum(1 for p in preds if p.is_total_correct)
        sport_perf[key] = {
            "total": total,
            "correct": correct,
            "accuracy": correct / max(total, 1),
            "ou_total": ou_total,
            "ou_correct": ou_correct,
            "ou_accuracy": ou_correct / max(ou_total, 1),
        }

    return templates.TemplateResponse(
        request,
        "performance.html",
        {
            "sport_perf": sport_perf,
            "sports": SPORTS,
        },
    )


@router.get("/daily")
async def daily_page(
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Daily picks page — games to play today and upcoming."""
    now = datetime.now(timezone.utc)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    today_end = today_start + timedelta(days=1)
    tomorrow_end = today_start + timedelta(days=2)
    week_end = today_start + timedelta(days=14)

    # Today's picks
    today_res = await session.execute(
        select(Prediction)
        .where(
            and_(
                Prediction.actual_outcome.is_(None),
                Prediction.match_date >= today_start,
                Prediction.match_date < today_end,
            )
        )
        .order_by(Prediction.match_date)
    )
    today_picks = today_res.scalars().all()

    # Tomorrow's picks
    tomorrow_res = await session.execute(
        select(Prediction)
        .where(
            and_(
                Prediction.actual_outcome.is_(None),
                Prediction.match_date >= today_end,
                Prediction.match_date < tomorrow_end,
            )
        )
        .order_by(Prediction.match_date)
    )
    tomorrow_picks = tomorrow_res.scalars().all()

    # Upcoming this week (day after tomorrow → 7 days out)
    upcoming_res = await session.execute(
        select(Prediction)
        .where(
            and_(
                Prediction.actual_outcome.is_(None),
                Prediction.match_date >= tomorrow_end,
                Prediction.match_date < week_end,
            )
        )
        .order_by(Prediction.match_date)
    )
    upcoming_picks = upcoming_res.scalars().all()

    # Group by sport
    def group_by_sport(picks):
        grouped = {}
        for p in picks:
            grouped.setdefault(p.sport, []).append(p)
        return grouped

    # Group upcoming by date then sport
    def group_by_date(picks):
        grouped = {}
        for p in picks:
            day_label = p.match_date.strftime("%A, %b %d")
            grouped.setdefault(day_label, []).append(p)
        return grouped

    return templates.TemplateResponse(
        request,
        "daily.html",
        {
            "today": group_by_sport(today_picks),
            "tomorrow": group_by_sport(tomorrow_picks),
            "upcoming": group_by_date(upcoming_picks),
            "today_count": len(today_picks),
            "tomorrow_count": len(tomorrow_picks),
            "upcoming_count": len(upcoming_picks),
            "today_date": today_start,
            "tomorrow_date": today_end,
            "sports": SPORTS,
        },
    )


@router.get("/bankroll")
async def bankroll_page(
    request: Request,
):
    """Bankroll tracker page."""
    return templates.TemplateResponse(
        request,
        "bankroll.html",
        {
            "sports": SPORTS,
        },
    )
