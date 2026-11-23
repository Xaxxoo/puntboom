"""Football (soccer) models."""

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from puntboom.models.base import Base, TimestampMixin


class FootballLeague(TimestampMixin, Base):
    __tablename__ = "football_leagues"

    id: Mapped[int] = mapped_column(primary_key=True)
    external_id: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(200))
    code: Mapped[str] = mapped_column(String(20))
    country: Mapped[str] = mapped_column(String(100))
    current_season: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    teams: Mapped[list["FootballTeam"]] = relationship(back_populates="league")
    matches: Mapped[list["FootballMatch"]] = relationship(back_populates="league")


class FootballTeam(TimestampMixin, Base):
    __tablename__ = "football_teams"

    id: Mapped[int] = mapped_column(primary_key=True)
    external_id: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(200))
    short_name: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    crest_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    league_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("football_leagues.id"), nullable=True
    )
    elo_rating: Mapped[float] = mapped_column(Float, default=1500.0)

    league: Mapped[Optional["FootballLeague"]] = relationship(back_populates="teams")


class FootballMatch(TimestampMixin, Base):
    __tablename__ = "football_matches"
    __table_args__ = (
        UniqueConstraint("external_id", name="uq_football_match_external"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    external_id: Mapped[str] = mapped_column(String(50), index=True)
    league_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("football_leagues.id"), nullable=True
    )
    home_team_id: Mapped[int] = mapped_column(ForeignKey("football_teams.id"))
    away_team_id: Mapped[int] = mapped_column(ForeignKey("football_teams.id"))
    match_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    status: Mapped[str] = mapped_column(String(20), default="SCHEDULED")
    matchday: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    season: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Scores
    home_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    away_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    home_ht_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    away_ht_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Result: H, D, A
    result: Mapped[Optional[str]] = mapped_column(String(1), nullable=True)

    league: Mapped[Optional["FootballLeague"]] = relationship(back_populates="matches")
    home_team: Mapped["FootballTeam"] = relationship(foreign_keys=[home_team_id])
    away_team: Mapped["FootballTeam"] = relationship(foreign_keys=[away_team_id])
