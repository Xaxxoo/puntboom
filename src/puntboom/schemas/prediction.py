"""Prediction Pydantic schemas."""

from datetime import datetime
from pydantic import BaseModel, ConfigDict


class PredictionSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    sport: str
    match_external_id: str
    match_date: datetime
    home_team: str
    away_team: str
    league: str | None = None
    prob_home: float
    prob_draw: float | None = None
    prob_away: float
    predicted_outcome: str
    confidence: float
    odds_home: float | None = None
    odds_draw: float | None = None
    odds_away: float | None = None
    is_value_bet: bool = False
    expected_value: float | None = None
    kelly_stake: float | None = None
    actual_outcome: str | None = None
    is_correct: bool | None = None
    total_line: float | None = None
    prob_over: float | None = None
    prob_under: float | None = None
    predicted_total: str | None = None
    odds_over: float | None = None
    odds_under: float | None = None
    actual_total: str | None = None
    is_total_correct: bool | None = None


class PredictionListResponse(BaseModel):
    predictions: list[PredictionSchema]
    total: int
    sport: str | None = None


class ModelPerformanceSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    sport: str
    model_version: str
    trained_at: datetime
    sample_count: int
    accuracy: float
    brier_score: float
    log_loss: float | None = None
    roi: float | None = None
