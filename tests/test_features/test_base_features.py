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
    scoring_std_dev,
    scoring_cv,
    over_pct,
    total_goals_avg,
    consecutive_over_streak,
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


# --- Tests for new totals helpers ---


class TestScoringStdDev:
    def test_known_values(self):
        # stdev([2, 4]) = sqrt(2) ≈ 1.4142
        result = scoring_std_dev([2, 4], 10)
        assert abs(result - 1.4142) < 0.01

    def test_constant_values(self):
        assert scoring_std_dev([3, 3, 3, 3], 10) == 0.0

    def test_empty(self):
        assert scoring_std_dev([], 10) == 0.0

    def test_single_element(self):
        assert scoring_std_dev([5], 10) == 0.0

    def test_window(self):
        # Only last 3 values: [1, 1, 1] → std = 0
        assert scoring_std_dev([10, 1, 1, 1], 3) == 0.0


class TestScoringCv:
    def test_known_values(self):
        # stdev([2, 4]) / mean([2, 4]) = 1.4142 / 3 ≈ 0.4714
        result = scoring_cv([2, 4], 10)
        assert abs(result - 0.4714) < 0.01

    def test_constant_values(self):
        assert scoring_cv([3, 3, 3], 10) == 0.0

    def test_empty(self):
        assert scoring_cv([], 10) == 0.0

    def test_single_element(self):
        assert scoring_cv([5], 10) == 0.0

    def test_all_zeros(self):
        # mean = 0, so CV should be 0 (division guard)
        assert scoring_cv([0, 0, 0], 10) == 0.0


class TestOverPct:
    def test_all_over(self):
        scored = [3, 3, 3]
        conceded = [2, 2, 2]
        # totals: 5, 5, 5 — all over 2.5
        assert over_pct(scored, conceded, 2.5, 10) == 1.0

    def test_none_over(self):
        scored = [0, 1, 0]
        conceded = [1, 0, 1]
        # totals: 1, 1, 1 — none over 2.5
        assert over_pct(scored, conceded, 2.5, 10) == 0.0

    def test_mixed(self):
        scored = [3, 1, 2]
        conceded = [1, 0, 2]
        # totals: 4, 1, 4 — 2/3 over 2.5
        assert abs(over_pct(scored, conceded, 2.5, 10) - 2 / 3) < 0.01

    def test_empty(self):
        assert over_pct([], [], 2.5, 10) == 0.5

    def test_window(self):
        scored = [0, 0, 3, 3]
        conceded = [0, 0, 3, 3]
        # window=2 → last 2: totals 6, 6 → all over 2.5
        assert over_pct(scored, conceded, 2.5, 2) == 1.0


class TestTotalGoalsAvg:
    def test_known(self):
        scored = [2, 1, 3]
        conceded = [1, 2, 0]
        # totals: 3, 3, 3 → avg 3.0
        assert total_goals_avg(scored, conceded, 10) == 3.0

    def test_empty(self):
        assert total_goals_avg([], [], 10) == 0.0

    def test_window(self):
        scored = [0, 2, 3]
        conceded = [0, 1, 1]
        # window=2 → last 2: totals 3, 4 → avg 3.5
        assert total_goals_avg(scored, conceded, 2) == 3.5


class TestConsecutiveOverStreak:
    def test_over_streak(self):
        scored = [1, 3, 3, 3]
        conceded = [0, 2, 2, 2]
        # totals: 1, 5, 5, 5 → last 3 are over 2.5 → +3
        assert consecutive_over_streak(scored, conceded, 2.5) == 3

    def test_under_streak(self):
        scored = [3, 1, 0, 1]
        conceded = [2, 1, 1, 0]
        # totals: 5, 2, 1, 1 → last 3 are under 2.5 → -3
        assert consecutive_over_streak(scored, conceded, 2.5) == -3

    def test_empty(self):
        assert consecutive_over_streak([], [], 2.5) == 0

    def test_single_over(self):
        assert consecutive_over_streak([3], [1], 2.5) == 1

    def test_single_under(self):
        assert consecutive_over_streak([1], [0], 2.5) == -1
