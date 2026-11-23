"""Matches API endpoints."""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from puntboom.database import get_session
from puntboom.models.football import FootballMatch, FootballTeam
from puntboom.models.basketball import BasketballGame, BasketballTeam
from puntboom.models.hockey import HockeyGame, HockeyTeam
from puntboom.models.tennis import TennisMatch, TennisPlayer

router = APIRouter(prefix="/api/matches", tags=["matches"])


@router.get("/{sport}")
async def get_matches(
    sport: str,
    status: str = Query("all", description="Filter: all, scheduled, finished"),
    limit: int = Query(50, ge=1, le=200),
    session: AsyncSession = Depends(get_session),
):
    """Get matches for a sport."""
    if sport == "football":
        return await _get_football_matches(session, status, limit)
    elif sport == "basketball":
        return await _get_basketball_matches(session, status, limit)
    elif sport == "hockey":
        return await _get_hockey_matches(session, status, limit)
    elif sport == "tennis":
        return await _get_tennis_matches(session, status, limit)
    return {"error": f"Unknown sport: {sport}"}


async def _get_football_matches(session: AsyncSession, status: str, limit: int):
    query = select(FootballMatch).options(
        selectinload(FootballMatch.home_team),
        selectinload(FootballMatch.away_team),
    )
    if status == "scheduled":
        query = query.where(FootballMatch.status == "SCHEDULED")
    elif status == "finished":
        query = query.where(FootballMatch.status == "FINISHED")
    query = query.order_by(FootballMatch.match_date.desc()).limit(limit)

    result = await session.execute(query)
    matches = result.scalars().all()

    return {
        "sport": "football",
        "matches": [
            {
                "id": m.id,
                "external_id": m.external_id,
                "home_team": m.home_team.name if m.home_team else "Unknown",
                "away_team": m.away_team.name if m.away_team else "Unknown",
                "match_date": m.match_date.isoformat() if m.match_date else None,
                "status": m.status,
                "home_score": m.home_score,
                "away_score": m.away_score,
                "result": m.result,
            }
            for m in matches
        ],
    }


async def _get_basketball_matches(session: AsyncSession, status: str, limit: int):
    query = select(BasketballGame).options(
        selectinload(BasketballGame.home_team),
        selectinload(BasketballGame.away_team),
    )
    if status == "scheduled":
        query = query.where(BasketballGame.status == "SCHEDULED")
    elif status == "finished":
        query = query.where(BasketballGame.status == "FINISHED")
    query = query.order_by(BasketballGame.game_date.desc()).limit(limit)

    result = await session.execute(query)
    games = result.scalars().all()

    return {
        "sport": "basketball",
        "matches": [
            {
                "id": g.id,
                "external_id": g.external_id,
                "home_team": g.home_team.name if g.home_team else "Unknown",
                "away_team": g.away_team.name if g.away_team else "Unknown",
                "match_date": g.game_date.isoformat() if g.game_date else None,
                "status": g.status,
                "home_score": g.home_score,
                "away_score": g.away_score,
                "result": g.result,
            }
            for g in games
        ],
    }


async def _get_hockey_matches(session: AsyncSession, status: str, limit: int):
    query = select(HockeyGame).options(
        selectinload(HockeyGame.home_team),
        selectinload(HockeyGame.away_team),
    )
    if status == "scheduled":
        query = query.where(HockeyGame.status == "SCHEDULED")
    elif status == "finished":
        query = query.where(HockeyGame.status == "FINISHED")
    query = query.order_by(HockeyGame.game_date.desc()).limit(limit)

    result = await session.execute(query)
    games = result.scalars().all()

    return {
        "sport": "hockey",
        "matches": [
            {
                "id": g.id,
                "external_id": g.external_id,
                "home_team": g.home_team.name if g.home_team else "Unknown",
                "away_team": g.away_team.name if g.away_team else "Unknown",
                "match_date": g.game_date.isoformat() if g.game_date else None,
                "status": g.status,
                "home_score": g.home_score,
                "away_score": g.away_score,
                "result": g.result,
            }
            for g in games
        ],
    }


async def _get_tennis_matches(session: AsyncSession, status: str, limit: int):
    query = select(TennisMatch).options(
        selectinload(TennisMatch.player1),
        selectinload(TennisMatch.player2),
    )
    if status == "scheduled":
        query = query.where(TennisMatch.status == "SCHEDULED")
    elif status == "finished":
        query = query.where(TennisMatch.status == "FINISHED")
    query = query.order_by(TennisMatch.match_date.desc()).limit(limit)

    result = await session.execute(query)
    matches = result.scalars().all()

    return {
        "sport": "tennis",
        "matches": [
            {
                "id": m.id,
                "external_id": m.external_id,
                "home_team": m.player1.name if m.player1 else "Unknown",
                "away_team": m.player2.name if m.player2 else "Unknown",
                "match_date": m.match_date.isoformat() if m.match_date else None,
                "tournament": m.tournament,
                "surface": m.surface,
                "status": m.status,
                "player1_sets": m.player1_sets,
                "player2_sets": m.player2_sets,
                "result": m.result,
            }
            for m in matches
        ],
    }
