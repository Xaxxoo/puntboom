"""Hockey-specific feature engineering."""

from datetime import datetime

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from puntboom.features.base import (
    build_base_features,
    implied_probability,
    goals_scored_avg,
    over_pct,
    consecutive_over_streak,
    total_goals_avg,
)
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

    # H2H history (was previously missing)
    h2h_games = [
        g
        for g in home_history
        if g.away_team_id == away_team.id or g.home_team_id == away_team.id
    ]
    h2h_results, _, _ = game_results_for_team(h2h_games, home_team.id)

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
        h2h_results=h2h_results,
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

    # --- Period scoring features (8) ---
    home_p1, home_p2, home_p3 = _period_scoring_for_team(home_history, home_team.id)
    away_p1, away_p2, away_p3 = _period_scoring_for_team(away_history, away_team.id)
    features["home_p1_avg"] = sum(home_p1) / max(len(home_p1), 1)
    features["home_p2_avg"] = sum(home_p2) / max(len(home_p2), 1)
    features["home_p3_avg"] = sum(home_p3) / max(len(home_p3), 1)
    features["away_p1_avg"] = sum(away_p1) / max(len(away_p1), 1)
    features["away_p2_avg"] = sum(away_p2) / max(len(away_p2), 1)
    features["away_p3_avg"] = sum(away_p3) / max(len(away_p3), 1)
    # 3rd period total (both teams combined) — late-game scoring tendency
    home_p3_totals = _period_total(home_history, 3)
    away_p3_totals = _period_total(away_history, 3)
    features["home_p3_total_avg"] = sum(home_p3_totals) / max(len(home_p3_totals), 1)
    features["away_p3_total_avg"] = sum(away_p3_totals) / max(len(away_p3_totals), 1)

    # --- Shots/pace features (5) ---
    home_total_shots = _avg_total_shots(home_history)
    away_total_shots = _avg_total_shots(away_history)
    features["home_total_shots_avg"] = home_total_shots
    features["away_total_shots_avg"] = away_total_shots
    features["combined_shots_avg"] = home_total_shots + away_total_shots
    features["home_shooting_pct"] = (
        features["home_goals_per_game"] / max(features["home_shots_avg"], 1)
    )
    features["away_shooting_pct"] = (
        features["away_goals_per_game"] / max(features["away_shots_avg"], 1)
    )

    # --- Multi-threshold over percentages (7) ---
    features["home_over45_pct"] = over_pct(home_scored, home_conceded, 4.5)
    features["away_over45_pct"] = over_pct(away_scored, away_conceded, 4.5)
    features["home_over55_pct"] = over_pct(home_scored, home_conceded, 5.5)
    features["away_over55_pct"] = over_pct(away_scored, away_conceded, 5.5)
    features["home_over65_pct"] = over_pct(home_scored, home_conceded, 6.5)
    features["away_over65_pct"] = over_pct(away_scored, away_conceded, 6.5)
    features["combined_over55_pct"] = (
        over_pct(home_scored, home_conceded, 5.5)
        + over_pct(away_scored, away_conceded, 5.5)
    ) / 2.0

    # --- Game type features (4) ---
    features["home_ot_rate"] = _overtime_rate(home_history)
    features["away_ot_rate"] = _overtime_rate(away_history)
    features["home_one_goal_pct"] = _one_goal_game_pct(home_history)
    features["away_one_goal_pct"] = _one_goal_game_pct(away_history)

    # --- Over 5.5 streaks (2) ---
    features["home_over55_streak"] = float(
        consecutive_over_streak(home_scored, home_conceded, 5.5)
    )
    features["away_over55_streak"] = float(
        consecutive_over_streak(away_scored, away_conceded, 5.5)
    )

    # --- PP opportunities (3) ---
    features["home_pp_opps_avg"] = _avg_pp_opportunities(home_history, home_team.id)
    features["away_pp_opps_avg"] = _avg_pp_opportunities(away_history, away_team.id)
    features["combined_pp_opps"] = (
        features["home_pp_opps_avg"] + features["away_pp_opps_avg"]
    )

    # --- H2H totals (2) ---
    h2h_avg, h2h_o55 = _h2h_total_goals(h2h_games)
    features["h2h_avg_total"] = h2h_avg
    features["h2h_over55_pct"] = h2h_o55

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


def _period_scoring_for_team(
    games: list[HockeyGame], team_id: int
) -> tuple[list[int], list[int], list[int]]:
    """Extract per-period goals for a team. Returns (p1, p2, p3) lists."""
    p1: list[int] = []
    p2: list[int] = []
    p3: list[int] = []
    for g in games:
        if g.home_team_id == team_id:
            if g.home_p1 is not None:
                p1.append(g.home_p1)
            if g.home_p2 is not None:
                p2.append(g.home_p2)
            if g.home_p3 is not None:
                p3.append(g.home_p3)
        elif g.away_team_id == team_id:
            if g.away_p1 is not None:
                p1.append(g.away_p1)
            if g.away_p2 is not None:
                p2.append(g.away_p2)
            if g.away_p3 is not None:
                p3.append(g.away_p3)
    return p1, p2, p3


def _period_total(games: list[HockeyGame], period: int) -> list[int]:
    """Total goals (both teams) for a given period across games."""
    totals: list[int] = []
    for g in games:
        if period == 1 and g.home_p1 is not None and g.away_p1 is not None:
            totals.append(g.home_p1 + g.away_p1)
        elif period == 2 and g.home_p2 is not None and g.away_p2 is not None:
            totals.append(g.home_p2 + g.away_p2)
        elif period == 3 and g.home_p3 is not None and g.away_p3 is not None:
            totals.append(g.home_p3 + g.away_p3)
    return totals


def _avg_total_shots(games: list[HockeyGame]) -> float:
    """Average total shots (both teams) per game."""
    totals = []
    for g in games:
        if g.home_shots is not None and g.away_shots is not None:
            totals.append(g.home_shots + g.away_shots)
    return sum(totals) / max(len(totals), 1)


def _overtime_rate(games: list[HockeyGame]) -> float:
    """Percentage of games going to overtime or shootout."""
    finished = [g for g in games if g.home_score is not None and g.away_score is not None]
    if not finished:
        return 0.0
    ot = sum(1 for g in finished if g.overtime or g.shootout)
    return ot / len(finished)


def _one_goal_game_pct(games: list[HockeyGame]) -> float:
    """Percentage of games decided by <= 1 goal."""
    finished = [g for g in games if g.home_score is not None and g.away_score is not None]
    if not finished:
        return 0.5
    close = sum(1 for g in finished if abs(g.home_score - g.away_score) <= 1)
    return close / len(finished)


def _avg_pp_opportunities(games: list[HockeyGame], team_id: int) -> float:
    """Average power play opportunities per game for a team."""
    opps: list[int] = []
    for g in games:
        if g.home_team_id == team_id and g.home_pp_opportunities is not None:
            opps.append(g.home_pp_opportunities)
        elif g.away_team_id == team_id and g.away_pp_opportunities is not None:
            opps.append(g.away_pp_opportunities)
    return sum(opps) / max(len(opps), 1)


def _h2h_total_goals(
    h2h_games: list[HockeyGame],
) -> tuple[float, float]:
    """Average total goals and over-5.5 rate in H2H games."""
    finished = [
        g for g in h2h_games
        if g.home_score is not None and g.away_score is not None
    ]
    if not finished:
        return 5.5, 0.5  # neutral defaults
    totals = [g.home_score + g.away_score for g in finished]
    avg = sum(totals) / len(totals)
    o55 = sum(1 for t in totals if t > 5.5) / len(totals)
    return avg, o55
