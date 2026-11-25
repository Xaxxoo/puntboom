"""Shared feature engineering utilities."""

from datetime import datetime, timedelta, timezone
from statistics import stdev

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Totals-oriented helper functions
# ---------------------------------------------------------------------------


def scoring_std_dev(scores: list[int], window: int = 10) -> float:
    """Standard deviation of scores over last N games."""
    if len(scores) < 2:
        return 0.0
    recent = scores[-window:]
    if len(recent) < 2:
        return 0.0
    return stdev(recent)


def scoring_cv(scores: list[int], window: int = 10) -> float:
    """Coefficient of variation (std/mean) of scores over last N games."""
    if len(scores) < 2:
        return 0.0
    recent = scores[-window:]
    if len(recent) < 2:
        return 0.0
    mean = sum(recent) / len(recent)
    if mean == 0:
        return 0.0
    return stdev(recent) / mean


def over_pct(
    scored: list[int], conceded: list[int], threshold: float, window: int = 10
) -> float:
    """Percentage of recent games where total goals exceeded threshold."""
    if not scored or not conceded:
        return 0.5
    n = min(len(scored), len(conceded), window)
    recent_scored = scored[-n:]
    recent_conceded = conceded[-n:]
    over = sum(
        1 for s, c in zip(recent_scored, recent_conceded) if (s + c) > threshold
    )
    return over / n


def total_goals_avg(
    scored: list[int], conceded: list[int], window: int = 10
) -> float:
    """Average total goals per game over recent games."""
    if not scored or not conceded:
        return 0.0
    n = min(len(scored), len(conceded), window)
    recent_scored = scored[-n:]
    recent_conceded = conceded[-n:]
    return sum(s + c for s, c in zip(recent_scored, recent_conceded)) / n


def consecutive_over_streak(
    scored: list[int], conceded: list[int], threshold: float
) -> int:
    """Current over/under streak. Positive = consecutive overs, negative = unders."""
    if not scored or not conceded:
        return 0
    n = min(len(scored), len(conceded))
    count = 0
    last_over = (scored[-1] + conceded[-1]) > threshold
    for s, c in zip(reversed(scored[-n:]), reversed(conceded[-n:])):
        is_over = (s + c) > threshold
        if is_over == last_over:
            count += 1
        else:
            break
    return count if last_over else -count


# ---------------------------------------------------------------------------
# Existing helpers
# ---------------------------------------------------------------------------


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
    # Strip timezone info to avoid naive/aware mismatch
    a = last_game_date.replace(tzinfo=None) if last_game_date.tzinfo else last_game_date
    b = current_date.replace(tzinfo=None) if current_date.tzinfo else current_date
    delta = b - a
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

    # --- Volatility features (6) ---
    features["home_goals_std"] = scoring_std_dev(team_scores)
    features["away_goals_std"] = scoring_std_dev(opp_scores)
    features["home_conceded_std"] = scoring_std_dev(team_conceded)
    features["away_conceded_std"] = scoring_std_dev(opp_conceded)
    features["home_goals_cv"] = scoring_cv(team_scores)
    features["away_goals_cv"] = scoring_cv(opp_scores)

    # --- Interaction features (5) ---
    home_avg = goals_scored_avg(team_scores)
    away_avg = goals_scored_avg(opp_scores)
    home_conc = goals_conceded_avg(team_conceded)
    away_conc = goals_conceded_avg(opp_conceded)
    features["combined_goals_avg"] = home_avg + away_avg
    features["combined_conceded_avg"] = home_conc + away_conc
    features["expected_total"] = (home_avg + away_conc + away_avg + home_conc) / 2.0
    features["attack_vs_defense_home"] = home_avg - away_conc  # negative = strong defense
    features["attack_vs_defense_away"] = away_avg - home_conc

    # --- Per-team totals (2) ---
    features["home_total_per_game"] = total_goals_avg(team_scores, team_conceded)
    features["away_total_per_game"] = total_goals_avg(opp_scores, opp_conceded)

    # --- Goal difference (2) ---
    features["home_goal_diff_avg"] = home_avg - home_conc
    features["away_goal_diff_avg"] = away_avg - away_conc

    return features
