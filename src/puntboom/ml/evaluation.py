"""Model evaluation metrics: accuracy, Brier score, calibration, ROI."""

import numpy as np


def evaluate_model(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    labels: list[str],
) -> dict:
    """Calculate evaluation metrics.

    Args:
        y_true: Array of true label indices.
        y_probs: Array of predicted probabilities (n_samples x n_classes).
        labels: List of class labels.

    Returns:
        Dictionary of metrics.
    """
    n_classes = len(labels)
    n_samples = len(y_true)

    if n_samples == 0:
        return {"accuracy": 0.0, "brier_score": 1.0, "log_loss": 10.0}

    # Accuracy
    predicted = np.argmax(y_probs, axis=1)
    accuracy = np.mean(predicted == y_true)

    # Brier score (multi-class)
    y_onehot = np.zeros((n_samples, n_classes))
    for i, label in enumerate(y_true):
        if 0 <= label < n_classes:
            y_onehot[i, label] = 1.0
    brier = np.mean(np.sum((y_probs - y_onehot) ** 2, axis=1))

    # Log loss
    eps = 1e-15
    clipped = np.clip(y_probs, eps, 1 - eps)
    log_loss_val = -np.mean(
        np.sum(y_onehot * np.log(clipped), axis=1)
    )

    # Calibration data (binned)
    calibration = {}
    for class_idx, label in enumerate(labels):
        probs = y_probs[:, class_idx]
        actual = (y_true == class_idx).astype(float)
        bins = np.linspace(0, 1, 11)
        bin_indices = np.digitize(probs, bins) - 1
        bin_means = []
        bin_true_means = []
        bin_counts = []
        for b in range(10):
            mask = bin_indices == b
            if mask.sum() > 0:
                bin_means.append(float(probs[mask].mean()))
                bin_true_means.append(float(actual[mask].mean()))
                bin_counts.append(int(mask.sum()))
            else:
                bin_means.append(float(bins[b] + 0.05))
                bin_true_means.append(0.0)
                bin_counts.append(0)
        calibration[label] = {
            "predicted": bin_means,
            "actual": bin_true_means,
            "counts": bin_counts,
        }

    return {
        "accuracy": float(accuracy),
        "brier_score": float(brier),
        "log_loss": float(log_loss_val),
        "calibration": calibration,
        "n_samples": n_samples,
    }


def calculate_roi(
    predictions: list[dict],
    min_ev: float = 0.0,
) -> dict:
    """Calculate ROI from resolved predictions.

    Each prediction dict should have: predicted_outcome, actual_outcome,
    odds (for the predicted outcome), confidence, kelly_stake.
    """
    total_staked = 0.0
    total_return = 0.0
    n_bets = 0
    n_wins = 0

    for pred in predictions:
        stake = pred.get("kelly_stake", 0.0)
        if stake <= 0:
            continue
        ev = pred.get("expected_value", 0.0)
        if ev < min_ev:
            continue

        odds = pred.get("odds", 0.0)
        is_correct = pred.get("is_correct", False)

        total_staked += stake
        n_bets += 1
        if is_correct:
            total_return += stake * odds
            n_wins += 1

    roi = (total_return - total_staked) / total_staked if total_staked > 0 else 0.0

    return {
        "total_staked": total_staked,
        "total_return": total_return,
        "profit": total_return - total_staked,
        "roi": roi,
        "n_bets": n_bets,
        "n_wins": n_wins,
        "win_rate": n_wins / max(n_bets, 1),
    }
