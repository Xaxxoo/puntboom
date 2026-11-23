"""Tests for Elo rating system."""

import pytest

from puntboom.features.elo import EloRatingSystem


def test_expected_score_equal_ratings():
    elo = EloRatingSystem()
    assert elo.expected_score(1500, 1500) == pytest.approx(0.5)


def test_expected_score_higher_rating():
    elo = EloRatingSystem()
    exp = elo.expected_score(1600, 1400)
    assert exp > 0.5
    assert exp < 1.0


def test_update_ratings_home_win():
    elo = EloRatingSystem(k_factor=32, home_advantage=0)
    new_home, new_away = elo.update_ratings(1500, 1500, 1.0)
    assert new_home > 1500
    assert new_away < 1500


def test_update_ratings_draw():
    elo = EloRatingSystem(k_factor=32, home_advantage=0)
    new_home, new_away = elo.update_ratings(1500, 1500, 0.5)
    # With no home advantage, draw should leave ratings unchanged
    assert new_home == pytest.approx(1500, abs=0.1)
    assert new_away == pytest.approx(1500, abs=0.1)


def test_update_ratings_away_win():
    elo = EloRatingSystem(k_factor=32, home_advantage=0)
    new_home, new_away = elo.update_ratings(1500, 1500, 0.0)
    assert new_home < 1500
    assert new_away > 1500


def test_win_probability_equal_ratings():
    elo = EloRatingSystem(home_advantage=0)
    probs = elo.win_probability(1500, 1500)
    assert probs["home"] + probs["draw"] + probs["away"] == pytest.approx(1.0, abs=0.01)


def test_win_probability_2way():
    elo = EloRatingSystem()
    probs = elo.win_probability_2way(1600, 1400)
    assert probs["home"] + probs["away"] == pytest.approx(1.0)
    assert probs["home"] > probs["away"]
