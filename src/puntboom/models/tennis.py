"""Tennis models."""

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Float, Integer, String, UniqueConstraint, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from puntboom.models.base import Base, TimestampMixin


class TennisPlayer(TimestampMixin, Base):
    __tablename__ = "tennis_players"

    id: Mapped[int] = mapped_column(primary_key=True)
    external_id: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(200))
    country: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    ranking: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    elo_rating: Mapped[float] = mapped_column(Float, default=1500.0)


class TennisMatch(TimestampMixin, Base):
    __tablename__ = "tennis_matches"
    __table_args__ = (
        UniqueConstraint("external_id", name="uq_tennis_match_external"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    external_id: Mapped[str] = mapped_column(String(50), index=True)
    player1_id: Mapped[int] = mapped_column(ForeignKey("tennis_players.id"))
    player2_id: Mapped[int] = mapped_column(ForeignKey("tennis_players.id"))
    match_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    tournament: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    surface: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    round: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="SCHEDULED")

    # Scores
    player1_sets: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    player2_sets: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    score_detail: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    is_grand_slam: Mapped[Optional[bool]] = mapped_column(default=False)

    # Result: 1 (player1 wins) or 2 (player2 wins)
    result: Mapped[Optional[str]] = mapped_column(String(1), nullable=True)

    player1: Mapped["TennisPlayer"] = relationship(foreign_keys=[player1_id])
    player2: Mapped["TennisPlayer"] = relationship(foreign_keys=[player2_id])
