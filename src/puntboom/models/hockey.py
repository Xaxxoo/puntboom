"""Ice Hockey (NHL) models."""

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from puntboom.models.base import Base, TimestampMixin


class HockeyTeam(TimestampMixin, Base):
    __tablename__ = "hockey_teams"

    id: Mapped[int] = mapped_column(primary_key=True)
    external_id: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(200))
    abbreviation: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    conference: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    division: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    elo_rating: Mapped[float] = mapped_column(Float, default=1500.0)


class HockeyGame(TimestampMixin, Base):
    __tablename__ = "hockey_games"
    __table_args__ = (
        UniqueConstraint("external_id", name="uq_hockey_game_external"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    external_id: Mapped[str] = mapped_column(String(50), index=True)
    home_team_id: Mapped[int] = mapped_column(ForeignKey("hockey_teams.id"))
    away_team_id: Mapped[int] = mapped_column(ForeignKey("hockey_teams.id"))
    game_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    season: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    game_type: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="SCHEDULED")

    # Scores
    home_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    away_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Period scores
    home_p1: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    home_p2: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    home_p3: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    away_p1: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    away_p2: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    away_p3: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Stats
    home_shots: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    away_shots: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    home_pp_goals: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    home_pp_opportunities: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    away_pp_goals: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    away_pp_opportunities: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    overtime: Mapped[Optional[bool]] = mapped_column(default=False)
    shootout: Mapped[Optional[bool]] = mapped_column(default=False)

    # Result: H or A
    result: Mapped[Optional[str]] = mapped_column(String(1), nullable=True)

    home_team: Mapped["HockeyTeam"] = relationship(foreign_keys=[home_team_id])
    away_team: Mapped["HockeyTeam"] = relationship(foreign_keys=[away_team_id])
