"""Tests for base feature engineering utilities."""

from datetime import datetime, timedelta, timezone

from puntboom.features.base import (
    rolling_win_pct,
    rolling_form,
    streak,
    rest_days,
    goals_scored_avg,
    head_to_head_record,
    implied_probability,
)


def test_rolling_win_pct():
    results = ["W", "W", "L", "W", "L"]
    assert rolling_win_pct(results, 5, "W") == 0.6


def test_rolling_win_pct_empty():
    assert rolling_win_pct([], 5, "W") == 0.5


def test_rolling_form_all_wins():
    results = ["W", "W", "W", "W", "W"]
    assert rolling_form(results, 5) == 1.0


def test_rolling_form_all_losses():
    results = ["L", "L", "L", "L", "L"]
    assert rolling_form(results, 5) == 0.0


def test_rolling_form_mixed():
    results = ["W", "D", "L", "W", "D"]
    # 3 + 1 + 0 + 3 + 1 = 8, max = 15
    assert rolling_form(results, 5) == 8 / 15


def test_streak_winning():
    results = ["L", "W", "W", "W"]
    assert streak(results) == 3


def test_streak_losing():
    results = ["W", "L", "L"]
    assert streak(results) == -2


def test_streak_empty():
    assert streak([]) == 0


def test_rest_days():
    now = datetime(2024, 1, 10, tzinfo=timezone.utc)
    last = datetime(2024, 1, 7, tzinfo=timezone.utc)
    assert rest_days(last, now) == 3


def test_rest_days_none():
    now = datetime(2024, 1, 10, tzinfo=timezone.utc)
    assert rest_days(None, now) == 7


def test_goals_scored_avg():
    scores = [2, 1, 3, 0, 4]
    assert goals_scored_avg(scores, 5) == 2.0


def test_head_to_head_record():
    results = ["W", "W", "L", "D", "W"]
    h2h = head_to_head_record(results)
    assert h2h["win_pct"] == 0.6
    assert h2h["draw_pct"] == 0.2
    assert h2h["loss_pct"] == 0.2


def test_implied_probability():
    assert implied_probability(2.0) == 0.5
    assert implied_probability(4.0) == 0.25
    assert implied_probability(0) == 0.0
