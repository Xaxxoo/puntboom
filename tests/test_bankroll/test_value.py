"""Tests for value bet detection."""

import pytest

from puntboom.bankroll.value import expected_value, is_value_bet, find_value_bets


def test_expected_value_positive():
    # 60% at odds 2.0: EV = 0.6 * 2.0 - 1 = 0.2
    ev = expected_value(0.6, 2.0)
    assert ev == pytest.approx(0.2)


def test_expected_value_negative():
    ev = expected_value(0.4, 2.0)
    assert ev == pytest.approx(-0.2)


def test_expected_value_zero():
    ev = expected_value(0.5, 2.0)
    assert ev == pytest.approx(0.0)


def test_is_value_bet_true():
    assert is_value_bet(0.6, 2.0, min_ev=0.05) is True


def test_is_value_bet_false():
    assert is_value_bet(0.4, 2.0, min_ev=0.05) is False


def test_find_value_bets():
    predictions = {"H": 0.5, "D": 0.3, "A": 0.2}
    odds = {"H": 2.5, "D": 3.0, "A": 6.0}
    bets = find_value_bets(predictions, odds, min_ev=0.05)
    # H: 0.5 * 2.5 - 1 = 0.25 (value)
    # A: 0.2 * 6.0 - 1 = 0.2 (value)
    # D: 0.3 * 3.0 - 1 = -0.1 (not value)
    assert len(bets) == 2
    assert bets[0]["outcome"] == "H"  # Highest EV
    assert bets[1]["outcome"] == "A"


def test_find_value_bets_none():
    predictions = {"H": 0.3, "A": 0.3}
    odds = {"H": 2.0, "A": 2.0}
    bets = find_value_bets(predictions, odds, min_ev=0.05)
    assert len(bets) == 0
