"""Basketball data collector using balldontlie.io v1 API."""

import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from puntboom.collectors.base import BaseCollector
from puntboom.config import settings
from puntboom.models.basketball import BasketballGame, BasketballTeam

logger = logging.getLogger(__name__)


class BasketballCollector(BaseCollector):
    BASE_URL = "https://api.balldontlie.io/v1"
    RATE_LIMIT = 5

    def __init__(self):
        super().__init__(api_key=settings.balldontlie_api_key)

    @property
    def headers(self) -> dict[str, str]:
        h = {"Accept": "application/json"}
        if self.api_key:
            h["Authorization"] = self.api_key
        return h

    async def collect_teams(self, session: AsyncSession) -> list[BasketballTeam]:
        """Fetch and store all NBA teams."""
        data = await self.get("/teams")
        teams_data = data.get("data", [])
        teams = []
        for t in teams_data:
            team = await self._upsert_team(session, t)
            teams.append(team)
        await session.commit()
        logger.info(f"Collected {len(teams)} basketball teams")
        return teams

    async def _upsert_team(
        self, session: AsyncSession, data: dict[str, Any]
    ) -> BasketballTeam:
        ext_id = str(data["id"])
        result = await session.execute(
            select(BasketballTeam).where(BasketballTeam.external_id == ext_id)
        )
        team = result.scalar_one_or_none()
        if team is None:
            team = BasketballTeam(
                external_id=ext_id,
                name=data.get("full_name", data.get("name", "Unknown")),
                abbreviation=data.get("abbreviation"),
                conference=data.get("conference"),
                division=data.get("division"),
            )
            session.add(team)
            await session.flush()
        return team

    async def collect_games(
        self,
        session: AsyncSession,
        season: int = 2024,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[BasketballGame]:
        """Fetch games, optionally filtered by date range."""
        all_games = []
        cursor = None
        while True:
            params: dict[str, Any] = {"seasons[]": season, "per_page": 100}
            if start_date:
                params["start_date"] = start_date
            if end_date:
                params["end_date"] = end_date
            if cursor:
                params["cursor"] = cursor

            data = await self.get("/games", params=params)
            games_data = data.get("data", [])

            for g in games_data:
                try:
                    game = await self._upsert_game(session, g)
                    all_games.append(game)
                except Exception:
                    logger.exception(f"Failed to process game {g.get('id')}")

            meta = data.get("meta", {})
            cursor = meta.get("next_cursor")
            if not cursor:
                break

        await session.commit()
        logger.info(f"Collected {len(all_games)} basketball games")
        return all_games

    async def _upsert_game(
        self, session: AsyncSession, data: dict[str, Any]
    ) -> BasketballGame:
        ext_id = str(data["id"])
        result = await session.execute(
            select(BasketballGame).where(BasketballGame.external_id == ext_id)
        )
        game = result.scalar_one_or_none()

        home_data = data.get("home_team", {})
        visitor_data = data.get("visitor_team", {})
        home_team = await self._upsert_team(session, home_data)
        away_team = await self._upsert_team(session, visitor_data)

        if game is None:
            game = BasketballGame(external_id=ext_id)
            session.add(game)

        game.home_team_id = home_team.id
        game.away_team_id = away_team.id
        game.season = data.get("season")

        date_str = data.get("date", "")
        if date_str:
            game.game_date = datetime.fromisoformat(
                date_str.replace("Z", "+00:00")
            )

        game.home_score = data.get("home_team_score")
        game.away_score = data.get("visitor_team_score")
        game.status = "FINISHED" if data.get("status") == "Final" else data.get("status", "SCHEDULED")

        if game.home_score is not None and game.away_score is not None and game.home_score + game.away_score > 0:
            game.result = "H" if game.home_score > game.away_score else "A"

        return game

    async def collect_all(self, session: AsyncSession) -> None:
        """Collect teams and recent games."""
        await self.collect_teams(session)
        await self.collect_games(session)
