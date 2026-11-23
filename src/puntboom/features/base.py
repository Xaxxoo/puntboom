"""Shared feature engineering utilities."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd


def rolling_win_pct(
    results: list[str], window: int = 5, win_value: str = "H"
) -> float:
    """Calculate rolling win percentage over last N games."""
    if not results:
        return 0.5
    recent = results[-window:]
    if not recent:
        return 0.5
    return sum(1 for r in recent if r == win_value) / len(recent)


def rolling_form(results: list[str], window: int = 5) -> float:
    """Calculate form points: win=3, draw=1, loss=0 (normalized)."""
    if not results:
        return 0.5
    recent = results[-window:]
    points = 0
    for r in recent:
        if r == "W":
            points += 3
        elif r == "D":
            points += 1
    max_points = len(recent) * 3
    return points / max_points if max_points > 0 else 0.5


def streak(results: list[str]) -> int:
    """Calculate current streak. Positive = wins, negative = losses, 0 = draws."""
    if not results:
        return 0
    current = results[-1]
    count = 0
    for r in reversed(results):
        if r == current:
            count += 1
        else:
            break
    if current == "W":
        return count
    elif current == "L":
        return -count
    return 0


def rest_days(last_game_date: datetime | None, current_date: datetime) -> int:
    """Calculate days since last game."""
    if last_game_date is None:
        return 7  # Default assumption
    delta = current_date - last_game_date
    return max(0, delta.days)


def goals_scored_avg(scores: list[int], window: int = 10) -> float:
    """Average goals/points scored over recent games."""
    if not scores:
        return 0.0
    recent = scores[-window:]
    return sum(recent) / len(recent)


def goals_conceded_avg(conceded: list[int], window: int = 10) -> float:
    """Average goals/points conceded over recent games."""
    if not conceded:
        return 0.0
    recent = conceded[-window:]
    return sum(recent) / len(recent)


def head_to_head_record(
    results: list[str], window: int = 10
) -> dict[str, float]:
    """H2H record from perspective of team 1. Returns win%, draw%, loss%."""
    if not results:
        return {"win_pct": 0.5, "draw_pct": 0.0, "loss_pct": 0.5}
    recent = results[-window:]
    n = len(recent)
    wins = sum(1 for r in recent if r == "W") / n
    draws = sum(1 for r in recent if r == "D") / n
    losses = sum(1 for r in recent if r == "L") / n
    return {"win_pct": wins, "draw_pct": draws, "loss_pct": losses}


def implied_probability(decimal_odds: float) -> float:
    """Convert decimal odds to implied probability."""
    if decimal_odds <= 0:
        return 0.0
    return 1.0 / decimal_odds


def build_base_features(
    team_results: list[str],
    team_scores: list[int],
    team_conceded: list[int],
    opp_results: list[str],
    opp_scores: list[int],
    opp_conceded: list[int],
    last_game_date: datetime | None,
    opp_last_game_date: datetime | None,
    current_date: datetime,
    elo_home: float = 1500.0,
    elo_away: float = 1500.0,
    h2h_results: list[str] | None = None,
) -> dict[str, float]:
    """Build a base feature dictionary common to all sports."""
    features = {
        "home_form_5": rolling_form(team_results, 5),
        "home_form_10": rolling_form(team_results, 10),
        "away_form_5": rolling_form(opp_results, 5),
        "away_form_10": rolling_form(opp_results, 10),
        "home_win_pct_5": rolling_win_pct(team_results, 5, "W"),
        "away_win_pct_5": rolling_win_pct(opp_results, 5, "W"),
        "home_streak": float(streak(team_results)),
        "away_streak": float(streak(opp_results)),
        "home_goals_avg": goals_scored_avg(team_scores),
        "home_conceded_avg": goals_conceded_avg(team_conceded),
        "away_goals_avg": goals_scored_avg(opp_scores),
        "away_conceded_avg": goals_conceded_avg(opp_conceded),
        "home_rest_days": float(rest_days(last_game_date, current_date)),
        "away_rest_days": float(rest_days(opp_last_game_date, current_date)),
        "elo_home": elo_home,
        "elo_away": elo_away,
        "elo_diff": elo_home - elo_away,
    }

    if h2h_results:
        h2h = head_to_head_record(h2h_results)
        features["h2h_win_pct"] = h2h["win_pct"]
        features["h2h_draw_pct"] = h2h["draw_pct"]
    else:
        features["h2h_win_pct"] = 0.5
        features["h2h_draw_pct"] = 0.0

    return features
