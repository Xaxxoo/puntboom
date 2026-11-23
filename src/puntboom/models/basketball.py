"""Basketball (NBA) models."""

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from puntboom.models.base import Base, TimestampMixin


class BasketballTeam(TimestampMixin, Base):
    __tablename__ = "basketball_teams"

    id: Mapped[int] = mapped_column(primary_key=True)
    external_id: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(200))
    abbreviation: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    conference: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    division: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    elo_rating: Mapped[float] = mapped_column(Float, default=1500.0)


class BasketballGame(TimestampMixin, Base):
    __tablename__ = "basketball_games"
    __table_args__ = (
        UniqueConstraint("external_id", name="uq_basketball_game_external"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    external_id: Mapped[str] = mapped_column(String(50), index=True)
    home_team_id: Mapped[int] = mapped_column(ForeignKey("basketball_teams.id"))
    away_team_id: Mapped[int] = mapped_column(ForeignKey("basketball_teams.id"))
    game_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    season: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="SCHEDULED")

    # Scores
    home_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    away_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Quarter scores
    home_q1: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    home_q2: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    home_q3: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    home_q4: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    away_q1: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    away_q2: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    away_q3: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    away_q4: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Result: H or A (no draws in NBA)
    result: Mapped[Optional[str]] = mapped_column(String(1), nullable=True)

    home_team: Mapped["BasketballTeam"] = relationship(foreign_keys=[home_team_id])
    away_team: Mapped["BasketballTeam"] = relationship(foreign_keys=[away_team_id])
