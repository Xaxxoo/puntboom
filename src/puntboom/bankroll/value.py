"""Value bet identification."""

from puntboom.config import settings


def expected_value(prob: float, odds: float) -> float:
    """Calculate expected value of a bet.

    EV = prob * decimal_odds - 1
    Positive EV means the bet has value.
    """
    if odds <= 0 or prob <= 0:
        return -1.0
    return prob * odds - 1.0


def is_value_bet(
    prob: float, odds: float, min_ev: float | None = None
) -> bool:
    """Check if a bet offers positive expected value above threshold."""
    if min_ev is None:
        min_ev = settings.min_value_threshold
    return expected_value(prob, odds) > min_ev


def find_value_bets(
    predictions: dict[str, float],
    odds: dict[str, float],
    min_ev: float | None = None,
) -> list[dict]:
    """Find value bets from prediction probabilities and odds.

    Args:
        predictions: Dict mapping outcomes to probabilities (e.g., {"H": 0.5, "D": 0.3, "A": 0.2}).
        odds: Dict mapping outcomes to decimal odds (e.g., {"H": 2.0, "D": 3.5, "A": 4.0}).
        min_ev: Minimum expected value threshold.

    Returns:
        List of value bet dicts sorted by EV descending.
    """
    if min_ev is None:
        min_ev = settings.min_value_threshold

    value_bets = []
    for outcome, prob in predictions.items():
        if outcome not in odds:
            continue
        ev = expected_value(prob, odds[outcome])
        if ev > min_ev:
            value_bets.append(
                {
                    "outcome": outcome,
                    "probability": prob,
                    "odds": odds[outcome],
                    "expected_value": ev,
                }
            )

    value_bets.sort(key=lambda x: x["expected_value"], reverse=True)
    return value_bets
