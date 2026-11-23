"""Kelly criterion for optimal bet sizing."""

from puntboom.config import settings


def kelly_criterion(prob: float, odds: float) -> float:
    """Calculate full Kelly stake as fraction of bankroll.

    Args:
        prob: Estimated probability of winning.
        odds: Decimal odds offered.

    Returns:
        Optimal fraction of bankroll to bet (can be negative = don't bet).
    """
    if odds <= 1.0 or prob <= 0.0 or prob >= 1.0:
        return 0.0
    b = odds - 1.0  # Net odds (profit per unit staked)
    q = 1.0 - prob
    kelly = (b * prob - q) / b
    return kelly


def fractional_kelly(
    prob: float, odds: float, fraction: float | None = None
) -> float:
    """Calculate fractional Kelly stake.

    Using a fraction (e.g., 0.25) of the full Kelly to reduce variance.
    """
    if fraction is None:
        fraction = settings.kelly_fraction
    full = kelly_criterion(prob, odds)
    if full <= 0:
        return 0.0
    return full * fraction


def kelly_stake_amount(
    prob: float, odds: float, bankroll: float, fraction: float | None = None
) -> float:
    """Calculate actual stake amount in currency units."""
    frac = fractional_kelly(prob, odds, fraction)
    return max(0.0, bankroll * frac)
