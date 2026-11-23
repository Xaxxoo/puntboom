"""Bankroll Pydantic schemas."""

from pydantic import BaseModel


class BetRecommendation(BaseModel):
    sport: str
    match: str
    match_date: str
    outcome: str
    probability: float
    odds: float
    expected_value: float
    kelly_fraction: float
    stake_amount: float


class BankrollSummary(BaseModel):
    bankroll: float
    total_staked: float
    total_return: float
    profit: float
    roi: float
    total_predictions: int
    correct: int
    accuracy: float


class PnLEntry(BaseModel):
    date: str
    sport: str
    match: str
    predicted: str
    actual: str
    stake: float
    profit: float
    cumulative_pnl: float
    bankroll_value: float
