"""Football-specific feature engineering."""

from datetime import datetime

import pandas as pd
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from puntboom.features.base import (
    build_base_features,
    implied_probability,
    goals_scored_avg,
)
from puntboom.features.elo import EloRatingSystem
from puntboom.models.football import FootballMatch, FootballTeam

elo_system = EloRatingSystem(k_factor=20.0, home_advantage=65.0)


async def get_team_history(
    session: AsyncSession, team_id: int, before_date: datetime, limit: int = 20
) -> list[FootballMatch]:
    """Get recent matches for a team before a date."""
    result = await session.execute(
        select(FootballMatch)
        .where(
            and_(
                FootballMatch.status == "FINISHED",
                FootballMatch.match_date < before_date,
                (FootballMatch.home_team_id == team_id)
                | (FootballMatch.away_team_id == team_id),
            )
        )
        .order_by(FootballMatch.match_date.desc())
        .limit(limit)
    )
    return list(result.scalars().all())


def match_results_for_team(
    matches: list[FootballMatch], team_id: int
) -> tuple[list[str], list[int], list[int]]:
    """Extract results, scores, and conceded from team perspective."""
    results = []
    scored = []
    conceded = []
    for m in reversed(matches):
        if m.home_score is None or m.away_score is None:
            continue
        if m.home_team_id == team_id:
            scored.append(m.home_score)
            conceded.append(m.away_score)
            if m.result == "H":
                results.append("W")
            elif m.result == "A":
                results.append("L")
            else:
                results.append("D")
        else:
            scored.append(m.away_score)
            conceded.append(m.home_score)
            if m.result == "A":
                results.append("W")
            elif m.result == "H":
                results.append("L")
            else:
                results.append("D")
    return results, scored, conceded


async def build_football_features(
    session: AsyncSession,
    home_team: FootballTeam,
    away_team: FootballTeam,
    match_date: datetime,
    odds_home: float | None = None,
    odds_draw: float | None = None,
    odds_away: float | None = None,
) -> dict[str, float]:
    """Build full feature set for a football match."""
    home_history = await get_team_history(session, home_team.id, match_date)
    away_history = await get_team_history(session, away_team.id, match_date)

    home_results, home_scored, home_conceded = match_results_for_team(
        home_history, home_team.id
    )
    away_results, away_scored, away_conceded = match_results_for_team(
        away_history, away_team.id
    )

    # H2H history
    h2h_matches = [
        m
        for m in home_history
        if m.away_team_id == away_team.id or m.home_team_id == away_team.id
    ]
    h2h_results, _, _ = match_results_for_team(h2h_matches, home_team.id)

    last_home_date = home_history[0].match_date if home_history else None
    last_away_date = away_history[0].match_date if away_history else None

    features = build_base_features(
        team_results=home_results,
        team_scores=home_scored,
        team_conceded=home_conceded,
        opp_results=away_results,
        opp_scores=away_scored,
        opp_conceded=away_conceded,
        last_game_date=last_home_date,
        opp_last_game_date=last_away_date,
        current_date=match_date,
        elo_home=home_team.elo_rating,
        elo_away=away_team.elo_rating,
        h2h_results=h2h_results,
    )

    # Football-specific features
    features["home_btts_pct"] = _btts_percentage(home_history, home_team.id)
    features["away_btts_pct"] = _btts_percentage(away_history, away_team.id)
    features["home_over25_pct"] = _over_goals_pct(home_history, 2.5)
    features["away_over25_pct"] = _over_goals_pct(away_history, 2.5)
    features["home_clean_sheet_pct"] = _clean_sheet_pct(home_history, home_team.id)
    features["away_clean_sheet_pct"] = _clean_sheet_pct(away_history, away_team.id)

    # Home-only form
    home_at_home = [m for m in home_history if m.home_team_id == home_team.id]
    away_at_away = [m for m in away_history if m.away_team_id == away_team.id]
    h_res, h_sc, _ = match_results_for_team(home_at_home, home_team.id)
    a_res, a_sc, _ = match_results_for_team(away_at_away, away_team.id)
    features["home_home_win_pct"] = sum(1 for r in h_res if r == "W") / max(len(h_res), 1)
    features["away_away_win_pct"] = sum(1 for r in a_res if r == "W") / max(len(a_res), 1)
    features["home_home_goals_avg"] = goals_scored_avg(h_sc, 5)
    features["away_away_goals_avg"] = goals_scored_avg(a_sc, 5)

    # Odds-implied probabilities
    if odds_home and odds_home > 0:
        features["implied_prob_home"] = implied_probability(odds_home)
    else:
        features["implied_prob_home"] = 0.0
    if odds_draw and odds_draw > 0:
        features["implied_prob_draw"] = implied_probability(odds_draw)
    else:
        features["implied_prob_draw"] = 0.0
    if odds_away and odds_away > 0:
        features["implied_prob_away"] = implied_probability(odds_away)
    else:
        features["implied_prob_away"] = 0.0

    return features


def _btts_percentage(matches: list[FootballMatch], team_id: int) -> float:
    """Both teams to score percentage."""
    finished = [m for m in matches if m.home_score is not None and m.away_score is not None]
    if not finished:
        return 0.5
    btts = sum(1 for m in finished if m.home_score > 0 and m.away_score > 0)
    return btts / len(finished)


def _over_goals_pct(matches: list[FootballMatch], threshold: float) -> float:
    """Percentage of matches with total goals over threshold."""
    finished = [m for m in matches if m.home_score is not None and m.away_score is not None]
    if not finished:
        return 0.5
    over = sum(
        1 for m in finished if (m.home_score + m.away_score) > threshold
    )
    return over / len(finished)


def _clean_sheet_pct(matches: list[FootballMatch], team_id: int) -> float:
    """Clean sheet percentage for a team."""
    finished = [m for m in matches if m.home_score is not None and m.away_score is not None]
    if not finished:
        return 0.0
    clean = 0
    for m in finished:
        if m.home_team_id == team_id and m.away_score == 0:
            clean += 1
        elif m.away_team_id == team_id and m.home_score == 0:
            clean += 1
    return clean / len(finished)
