"""Tennis-specific feature engineering."""

from datetime import datetime

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from puntboom.features.base import (
    build_base_features,
    implied_probability,
    rolling_win_pct,
    rest_days,
)
from puntboom.features.elo import EloRatingSystem
from puntboom.models.tennis import TennisMatch, TennisPlayer

elo_system = EloRatingSystem(k_factor=32.0, home_advantage=0.0)

SURFACE_MAP = {"hard": 0, "clay": 1, "grass": 2, "carpet": 3}
ROUND_ORDER = {
    "Final": 7,
    "Semifinals": 6,
    "Quarterfinals": 5,
    "Round of 16": 4,
    "Round of 32": 3,
    "Round of 64": 2,
    "Round of 128": 1,
    "Qualifying": 0,
}


async def get_player_history(
    session: AsyncSession, player_id: int, before_date: datetime, limit: int = 20
) -> list[TennisMatch]:
    result = await session.execute(
        select(TennisMatch)
        .where(
            and_(
                TennisMatch.status == "FINISHED",
                TennisMatch.match_date < before_date,
                (TennisMatch.player1_id == player_id)
                | (TennisMatch.player2_id == player_id),
            )
        )
        .order_by(TennisMatch.match_date.desc())
        .limit(limit)
    )
    return list(result.scalars().all())


def match_results_for_player(
    matches: list[TennisMatch], player_id: int
) -> tuple[list[str], list[int], list[int]]:
    results = []
    sets_won = []
    sets_lost = []
    for m in reversed(matches):
        if m.player1_id == player_id:
            results.append("W" if m.result == "1" else "L")
            sets_won.append(m.player1_sets or 0)
            sets_lost.append(m.player2_sets or 0)
        else:
            results.append("W" if m.result == "2" else "L")
            sets_won.append(m.player2_sets or 0)
            sets_lost.append(m.player1_sets or 0)
    return results, sets_won, sets_lost


def surface_win_pct(
    matches: list[TennisMatch], player_id: int, surface: str | None
) -> float:
    """Win percentage on a specific surface."""
    if not surface:
        return 0.5
    surface_matches = [m for m in matches if m.surface and m.surface.lower() == surface.lower()]
    if not surface_matches:
        return 0.5
    wins = 0
    for m in surface_matches:
        if (m.player1_id == player_id and m.result == "1") or (
            m.player2_id == player_id and m.result == "2"
        ):
            wins += 1
    return wins / len(surface_matches)


async def build_tennis_features(
    session: AsyncSession,
    player1: TennisPlayer,
    player2: TennisPlayer,
    match_date: datetime,
    surface: str | None = None,
    round_name: str | None = None,
    is_grand_slam: bool = False,
    odds_p1: float | None = None,
    odds_p2: float | None = None,
) -> dict[str, float]:
    """Build full feature set for a tennis match."""
    p1_history = await get_player_history(session, player1.id, match_date)
    p2_history = await get_player_history(session, player2.id, match_date)

    p1_results, p1_sets_w, p1_sets_l = match_results_for_player(
        p1_history, player1.id
    )
    p2_results, p2_sets_w, p2_sets_l = match_results_for_player(
        p2_history, player2.id
    )

    last_p1_date = p1_history[0].match_date if p1_history else None
    last_p2_date = p2_history[0].match_date if p2_history else None

    features = build_base_features(
        team_results=p1_results,
        team_scores=p1_sets_w,
        team_conceded=p1_sets_l,
        opp_results=p2_results,
        opp_scores=p2_sets_w,
        opp_conceded=p2_sets_l,
        last_game_date=last_p1_date,
        opp_last_game_date=last_p2_date,
        current_date=match_date,
        elo_home=player1.elo_rating,
        elo_away=player2.elo_rating,
    )

    # Tennis-specific features
    features["p1_ranking"] = float(player1.ranking or 100)
    features["p2_ranking"] = float(player2.ranking or 100)
    features["ranking_diff"] = features["p1_ranking"] - features["p2_ranking"]

    # Surface performance
    features["p1_surface_win_pct"] = surface_win_pct(p1_history, player1.id, surface)
    features["p2_surface_win_pct"] = surface_win_pct(p2_history, player2.id, surface)

    # Surface encoding
    features["surface_encoded"] = float(SURFACE_MAP.get(surface.lower() if surface else "", 0))

    # Round importance
    features["round_encoded"] = float(ROUND_ORDER.get(round_name or "", 3))

    # Grand slam flag
    features["is_grand_slam"] = 1.0 if is_grand_slam else 0.0

    # Fatigue: matches in last 7 days
    from datetime import timedelta

    week_ago = match_date - timedelta(days=7)
    p1_recent = [m for m in p1_history if m.match_date and m.match_date >= week_ago]
    p2_recent = [m for m in p2_history if m.match_date and m.match_date >= week_ago]
    features["p1_matches_7d"] = float(len(p1_recent))
    features["p2_matches_7d"] = float(len(p2_recent))

    # Sets ratio in recent matches
    features["p1_sets_ratio"] = sum(p1_sets_w) / max(sum(p1_sets_w) + sum(p1_sets_l), 1)
    features["p2_sets_ratio"] = sum(p2_sets_w) / max(sum(p2_sets_w) + sum(p2_sets_l), 1)

    # Odds
    if odds_p1 and odds_p1 > 0:
        features["implied_prob_p1"] = implied_probability(odds_p1)
    else:
        features["implied_prob_p1"] = 0.0
    if odds_p2 and odds_p2 > 0:
        features["implied_prob_p2"] = implied_probability(odds_p2)
    else:
        features["implied_prob_p2"] = 0.0

    return features
