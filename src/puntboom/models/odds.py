"""Odds snapshot model (cross-sport)."""

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Float, String
from sqlalchemy.orm import Mapped, mapped_column

from puntboom.models.base import Base, TimestampMixin


class OddsSnapshot(TimestampMixin, Base):
    __tablename__ = "odds_snapshots"

    id: Mapped[int] = mapped_column(primary_key=True)
    sport: Mapped[str] = mapped_column(String(20), index=True)
    match_external_id: Mapped[str] = mapped_column(String(100), index=True)
    bookmaker: Mapped[str] = mapped_column(String(100))
    market: Mapped[str] = mapped_column(String(50))
    snapshot_time: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    # Outcomes (up to 3 for football H/D/A, 2 for others)
    outcome1_label: Mapped[str] = mapped_column(String(100))
    outcome1_odds: Mapped[float] = mapped_column(Float)
    outcome2_label: Mapped[str] = mapped_column(String(100))
    outcome2_odds: Mapped[float] = mapped_column(Float)
    outcome3_label: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    outcome3_odds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
