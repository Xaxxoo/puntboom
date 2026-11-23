"""Tennis Pydantic schemas."""

from datetime import datetime
from pydantic import BaseModel, ConfigDict


class TennisPlayerSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    external_id: str
    name: str
    ranking: int | None = None
    elo_rating: float


class TennisMatchSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    external_id: str
    player1: str
    player2: str
    match_date: datetime
    tournament: str | None = None
    surface: str | None = None
    status: str
    player1_sets: int | None = None
    player2_sets: int | None = None
    result: str | None = None
