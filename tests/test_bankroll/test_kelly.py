"""Tests for Kelly criterion."""

import pytest

from puntboom.bankroll.kelly import kelly_criterion, fractional_kelly, kelly_stake_amount


def test_kelly_positive_edge():
    # 60% chance to win at even odds (2.0)
    kelly = kelly_criterion(0.6, 2.0)
    assert kelly == pytest.approx(0.2)


def test_kelly_no_edge():
    # 50% chance at even odds = no edge
    kelly = kelly_criterion(0.5, 2.0)
    assert kelly == pytest.approx(0.0)


def test_kelly_negative_edge():
    # 40% chance at even odds = negative
    kelly = kelly_criterion(0.4, 2.0)
    assert kelly < 0


def test_kelly_edge_cases():
    assert kelly_criterion(0.0, 2.0) == 0.0
    assert kelly_criterion(1.0, 2.0) == 0.0
    assert kelly_criterion(0.5, 1.0) == 0.0


def test_fractional_kelly():
    fk = fractional_kelly(0.6, 2.0, fraction=0.25)
    full = kelly_criterion(0.6, 2.0)
    assert fk == pytest.approx(full * 0.25)


def test_fractional_kelly_no_edge():
    fk = fractional_kelly(0.4, 2.0, fraction=0.25)
    assert fk == 0.0


def test_kelly_stake_amount():
    amount = kelly_stake_amount(0.6, 2.0, bankroll=1000, fraction=0.25)
    expected = 1000 * 0.2 * 0.25  # $50
    assert amount == pytest.approx(expected)
