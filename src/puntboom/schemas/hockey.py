"""Hockey Pydantic schemas."""

from datetime import datetime
from pydantic import BaseModel, ConfigDict


class HockeyTeamSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    external_id: str
    name: str
    abbreviation: str | None = None
    elo_rating: float


class HockeyGameSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    external_id: str
    home_team: str
    away_team: str
    game_date: datetime
    status: str
    home_score: int | None = None
    away_score: int | None = None
    result: str | None = None
