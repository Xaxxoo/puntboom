"""Basketball-specific feature engineering."""

from datetime import datetime, timedelta

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from puntboom.features.base import build_base_features, implied_probability
from puntboom.features.elo import EloRatingSystem
from puntboom.models.basketball import BasketballGame, BasketballTeam

elo_system = EloRatingSystem(k_factor=20.0, home_advantage=100.0)


async def get_team_history(
    session: AsyncSession, team_id: int, before_date: datetime, limit: int = 20
) -> list[BasketballGame]:
    result = await session.execute(
        select(BasketballGame)
        .where(
            and_(
                BasketballGame.status == "FINISHED",
                BasketballGame.game_date < before_date,
                (BasketballGame.home_team_id == team_id)
                | (BasketballGame.away_team_id == team_id),
            )
        )
        .order_by(BasketballGame.game_date.desc())
        .limit(limit)
    )
    return list(result.scalars().all())


def game_results_for_team(
    games: list[BasketballGame], team_id: int
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


def is_back_to_back(games: list[BasketballGame], team_id: int, game_date: datetime) -> bool:
    """Check if team played yesterday."""
    yesterday = game_date - timedelta(days=1)
    for g in games:
        if g.game_date and abs((g.game_date.date() - yesterday.date()).days) == 0:
            return True
    return False


async def build_basketball_features(
    session: AsyncSession,
    home_team: BasketballTeam,
    away_team: BasketballTeam,
    game_date: datetime,
    odds_home: float | None = None,
    odds_away: float | None = None,
) -> dict[str, float]:
    """Build full feature set for a basketball game."""
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

    # Basketball-specific
    features["home_ppg"] = sum(home_scored) / max(len(home_scored), 1)
    features["away_ppg"] = sum(away_scored) / max(len(away_scored), 1)
    features["home_opp_ppg"] = sum(home_conceded) / max(len(home_conceded), 1)
    features["away_opp_ppg"] = sum(away_conceded) / max(len(away_conceded), 1)

    # Net rating proxy (points diff per game)
    features["home_net_rating"] = features["home_ppg"] - features["home_opp_ppg"]
    features["away_net_rating"] = features["away_ppg"] - features["away_opp_ppg"]

    # Back-to-back
    features["home_b2b"] = 1.0 if is_back_to_back(home_history, home_team.id, game_date) else 0.0
    features["away_b2b"] = 1.0 if is_back_to_back(away_history, away_team.id, game_date) else 0.0

    # Pace proxy (total points per game)
    features["home_pace"] = features["home_ppg"] + features["home_opp_ppg"]
    features["away_pace"] = features["away_ppg"] + features["away_opp_ppg"]

    # Home/away specific
    home_at_home = [g for g in home_history if g.home_team_id == home_team.id]
    away_at_away = [g for g in away_history if g.away_team_id == away_team.id]
    h_res, _, _ = game_results_for_team(home_at_home, home_team.id)
    a_res, _, _ = game_results_for_team(away_at_away, away_team.id)
    features["home_home_win_pct"] = sum(1 for r in h_res if r == "W") / max(len(h_res), 1)
    features["away_away_win_pct"] = sum(1 for r in a_res if r == "W") / max(len(a_res), 1)

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
