"""Hockey-specific feature engineering."""

from datetime import datetime

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from puntboom.features.base import build_base_features, implied_probability
from puntboom.features.elo import EloRatingSystem
from puntboom.models.hockey import HockeyGame, HockeyTeam

elo_system = EloRatingSystem(k_factor=20.0, home_advantage=40.0)


async def get_team_history(
    session: AsyncSession, team_id: int, before_date: datetime, limit: int = 20
) -> list[HockeyGame]:
    result = await session.execute(
        select(HockeyGame)
        .where(
            and_(
                HockeyGame.status == "FINISHED",
                HockeyGame.game_date < before_date,
                (HockeyGame.home_team_id == team_id)
                | (HockeyGame.away_team_id == team_id),
            )
        )
        .order_by(HockeyGame.game_date.desc())
        .limit(limit)
    )
    return list(result.scalars().all())


def game_results_for_team(
    games: list[HockeyGame], team_id: int
) -> tuple[list[str], list[int], list[int]]:
    results = []
    scored = []
    conceded = []
    for g in reversed(games):
        if g.home_score is None or g.away_score is None:
            continue
        if g.home_team_id == team_id:
            scored.append(g.home_score)
            conceded.append(g.away_score)
            results.append("W" if g.result == "H" else "L")
        else:
            scored.append(g.away_score)
            conceded.append(g.home_score)
            results.append("W" if g.result == "A" else "L")
    return results, scored, conceded


async def build_hockey_features(
    session: AsyncSession,
    home_team: HockeyTeam,
    away_team: HockeyTeam,
    game_date: datetime,
    odds_home: float | None = None,
    odds_away: float | None = None,
) -> dict[str, float]:
    """Build full feature set for a hockey game."""
    home_history = await get_team_history(session, home_team.id, game_date)
    away_history = await get_team_history(session, away_team.id, game_date)

    home_results, home_scored, home_conceded = game_results_for_team(
        home_history, home_team.id
    )
    away_results, away_scored, away_conceded = game_results_for_team(
        away_history, away_team.id
    )

    last_home_date = home_history[0].game_date if home_history else None
    last_away_date = away_history[0].game_date if away_history else None

    features = build_base_features(
        team_results=home_results,
        team_scores=home_scored,
        team_conceded=home_conceded,
        opp_results=away_results,
        opp_scores=away_scored,
        opp_conceded=away_conceded,
        last_game_date=last_home_date,
        opp_last_game_date=last_away_date,
        current_date=game_date,
        elo_home=home_team.elo_rating,
        elo_away=away_team.elo_rating,
    )

    # Hockey-specific features
    features["home_goals_per_game"] = sum(home_scored) / max(len(home_scored), 1)
    features["away_goals_per_game"] = sum(away_scored) / max(len(away_scored), 1)
    features["home_goals_against_avg"] = sum(home_conceded) / max(len(home_conceded), 1)
    features["away_goals_against_avg"] = sum(away_conceded) / max(len(away_conceded), 1)

    # Power play percentage
    features["home_pp_pct"] = _power_play_pct(home_history, home_team.id)
    features["away_pp_pct"] = _power_play_pct(away_history, away_team.id)

    # Penalty kill (inverse of opponent's PP%)
    features["home_pk_pct"] = _penalty_kill_pct(home_history, home_team.id)
    features["away_pk_pct"] = _penalty_kill_pct(away_history, away_team.id)

    # Shots
    features["home_shots_avg"] = _avg_shots(home_history, home_team.id, is_home=True)
    features["away_shots_avg"] = _avg_shots(away_history, away_team.id, is_home=False)

    # Save percentage proxy
    home_shots_against = _avg_shots_against(home_history, home_team.id)
    away_shots_against = _avg_shots_against(away_history, away_team.id)
    features["home_save_pct"] = (
        1.0 - features["home_goals_against_avg"] / max(home_shots_against, 1)
        if home_shots_against > 0
        else 0.9
    )
    features["away_save_pct"] = (
        1.0 - features["away_goals_against_avg"] / max(away_shots_against, 1)
        if away_shots_against > 0
        else 0.9
    )

    # Odds
    if odds_home and odds_home > 0:
        features["implied_prob_home"] = implied_probability(odds_home)
    else:
        features["implied_prob_home"] = 0.0
    if odds_away and odds_away > 0:
        features["implied_prob_away"] = implied_probability(odds_away)
    else:
        features["implied_prob_away"] = 0.0

    return features


def _power_play_pct(games: list[HockeyGame], team_id: int) -> float:
    total_goals = 0
    total_opps = 0
    for g in games:
        if g.home_team_id == team_id:
            total_goals += g.home_pp_goals or 0
            total_opps += g.home_pp_opportunities or 0
        else:
            total_goals += g.away_pp_goals or 0
            total_opps += g.away_pp_opportunities or 0
    return total_goals / max(total_opps, 1)


def _penalty_kill_pct(games: list[HockeyGame], team_id: int) -> float:
    opp_goals = 0
    opp_opps = 0
    for g in games:
        if g.home_team_id == team_id:
            opp_goals += g.away_pp_goals or 0
            opp_opps += g.away_pp_opportunities or 0
        else:
            opp_goals += g.home_pp_goals or 0
            opp_opps += g.home_pp_opportunities or 0
    return 1.0 - (opp_goals / max(opp_opps, 1))


def _avg_shots(games: list[HockeyGame], team_id: int, is_home: bool = True) -> float:
    shots = []
    for g in games:
        if g.home_team_id == team_id and g.home_shots is not None:
            shots.append(g.home_shots)
        elif g.away_team_id == team_id and g.away_shots is not None:
            shots.append(g.away_shots)
    return sum(shots) / max(len(shots), 1)


def _avg_shots_against(games: list[HockeyGame], team_id: int) -> float:
    shots = []
    for g in games:
        if g.home_team_id == team_id and g.away_shots is not None:
            shots.append(g.away_shots)
        elif g.away_team_id == team_id and g.home_shots is not None:
            shots.append(g.home_shots)
    return sum(shots) / max(len(shots), 1)
