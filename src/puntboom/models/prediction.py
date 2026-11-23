"""Prediction and model performance tracking models."""

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Float, Integer, String, Text, Boolean
from sqlalchemy.orm import Mapped, mapped_column

from puntboom.models.base import Base, TimestampMixin


class Prediction(TimestampMixin, Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(primary_key=True)
    sport: Mapped[str] = mapped_column(String(20), index=True)
    match_external_id: Mapped[str] = mapped_column(String(100), index=True)
    match_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    home_team: Mapped[str] = mapped_column(String(200))
    away_team: Mapped[str] = mapped_column(String(200))
    league: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)

    # Probabilities
    prob_home: Mapped[float] = mapped_column(Float)
    prob_draw: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    prob_away: Mapped[float] = mapped_column(Float)

    # Predicted outcome
    predicted_outcome: Mapped[str] = mapped_column(String(10))
    confidence: Mapped[float] = mapped_column(Float)

    # Odds at prediction time
    odds_home: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    odds_draw: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    odds_away: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Value bet info
    is_value_bet: Mapped[bool] = mapped_column(Boolean, default=False)
    expected_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    kelly_stake: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Actual result (filled after match)
    actual_outcome: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    is_correct: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)

    # Per-team scoring averages (computed at prediction time)
    home_avg_scored: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    away_avg_scored: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    home_avg_conceded: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    away_avg_conceded: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Over/Under (Totals) market
    expected_goals: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    total_line: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    prob_over: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    prob_under: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    predicted_total: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    odds_over: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    odds_under: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    actual_total: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    is_total_correct: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)

    model_version: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)


class ModelPerformance(TimestampMixin, Base):
    __tablename__ = "model_performance"

    id: Mapped[int] = mapped_column(primary_key=True)
    sport: Mapped[str] = mapped_column(String(20), index=True)
    model_version: Mapped[str] = mapped_column(String(50))
    trained_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    sample_count: Mapped[int] = mapped_column(Integer)

    # Metrics
    accuracy: Mapped[float] = mapped_column(Float)
    brier_score: Mapped[float] = mapped_column(Float)
    log_loss: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    roi: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Serialized details
    details_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
