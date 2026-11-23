"""Football Pydantic schemas."""

from datetime import datetime
from pydantic import BaseModel, ConfigDict


class FootballTeamSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    external_id: str
    name: str
    short_name: str | None = None
    elo_rating: float


class FootballMatchSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    external_id: str
    home_team: str
    away_team: str
    match_date: datetime
    status: str
    home_score: int | None = None
    away_score: int | None = None
    result: str | None = None
