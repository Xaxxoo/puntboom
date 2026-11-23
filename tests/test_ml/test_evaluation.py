"""Tests for ML evaluation metrics."""

import numpy as np
import pytest

from puntboom.ml.evaluation import evaluate_model, calculate_roi


def test_evaluate_model_perfect():
    y_true = np.array([0, 1, 2, 0, 1])
    y_probs = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    metrics = evaluate_model(y_true, y_probs, ["H", "D", "A"])
    assert metrics["accuracy"] == 1.0
    assert metrics["brier_score"] == pytest.approx(0.0)


def test_evaluate_model_random():
    y_true = np.array([0, 1, 0, 1])
    y_probs = np.array([
        [0.5, 0.5],
        [0.5, 0.5],
        [0.5, 0.5],
        [0.5, 0.5],
    ])
    metrics = evaluate_model(y_true, y_probs, ["H", "A"])
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["brier_score"] > 0


def test_evaluate_model_empty():
    metrics = evaluate_model(np.array([]), np.array([]).reshape(0, 2), ["H", "A"])
    assert metrics["accuracy"] == 0.0


def test_calculate_roi_winning():
    predictions = [
        {"kelly_stake": 10.0, "odds": 2.0, "is_correct": True, "expected_value": 0.1},
        {"kelly_stake": 10.0, "odds": 2.0, "is_correct": True, "expected_value": 0.1},
    ]
    roi = calculate_roi(predictions)
    assert roi["n_bets"] == 2
    assert roi["n_wins"] == 2
    assert roi["profit"] == 20.0  # (10*2 + 10*2) - 20 = 20
    assert roi["roi"] == 1.0


def test_calculate_roi_losing():
    predictions = [
        {"kelly_stake": 10.0, "odds": 2.0, "is_correct": False, "expected_value": 0.1},
    ]
    roi = calculate_roi(predictions)
    assert roi["profit"] == -10.0
    assert roi["roi"] == -1.0
