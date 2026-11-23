"""Ice Hockey data collector using NHL Stats API (api-web.nhle.com)."""

import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from puntboom.collectors.base import BaseCollector
from puntboom.models.hockey import HockeyGame, HockeyTeam

logger = logging.getLogger(__name__)


class HockeyCollector(BaseCollector):
    BASE_URL = "https://api-web.nhle.com"
    RATE_LIMIT = 30

    def __init__(self):
        super().__init__()

    async def collect_teams(self, session: AsyncSession) -> list[HockeyTeam]:
        """Fetch and store all NHL teams."""
        data = await self.get("/v1/standings/now")
        standings = data.get("standings", [])
        teams = []
        seen = set()
        for entry in standings:
            abbrev = entry.get("teamAbbrev", {}).get("default", "")
            if abbrev in seen:
                continue
            seen.add(abbrev)
            team = await self._upsert_team(session, entry)
            teams.append(team)
        await session.commit()
        logger.info(f"Collected {len(teams)} hockey teams")
        return teams

    async def _upsert_team(
        self, session: AsyncSession, data: dict[str, Any]
    ) -> HockeyTeam:
        abbrev = data.get("teamAbbrev", {}).get("default", "")
        ext_id = abbrev
        result = await session.execute(
            select(HockeyTeam).where(HockeyTeam.external_id == ext_id)
        )
        team = result.scalar_one_or_none()
        if team is None:
            team = HockeyTeam(
                external_id=ext_id,
                name=data.get("teamName", {}).get("default", abbrev),
                abbreviation=abbrev,
                conference=data.get("conferenceName"),
                division=data.get("divisionName"),
            )
            session.add(team)
            await session.flush()
        return team

    async def collect_schedule(
        self, session: AsyncSession, date: str | None = None
    ) -> list[HockeyGame]:
        """Fetch schedule for a given week. date format: YYYY-MM-DD."""
        path = "/v1/schedule/now" if not date else f"/v1/schedule/{date}"
        data = await self.get(path)
        game_weeks = data.get("gameWeek", [])
        all_games = []

        for day in game_weeks:
            for g in day.get("games", []):
                try:
                    game = await self._upsert_game(session, g)
                    if game:
                        all_games.append(game)
                except Exception:
                    logger.exception(f"Failed to process game {g.get('id')}")

        await session.commit()
        logger.info(f"Collected {len(all_games)} hockey games")
        return all_games

    async def collect_game_detail(
        self, session: AsyncSession, game_id: int
    ) -> HockeyGame | None:
        """Fetch detailed box score for a finished game."""
        try:
            data = await self.get(f"/v1/gamecenter/{game_id}/boxscore")
        except Exception:
            logger.exception(f"Failed to fetch boxscore for game {game_id}")
            return None

        ext_id = str(game_id)
        result = await session.execute(
            select(HockeyGame).where(HockeyGame.external_id == ext_id)
        )
        game = result.scalar_one_or_none()
        if game is None:
            return None

        # Update with boxscore stats
        home_stats = data.get("homeTeam", {})
        away_stats = data.get("awayTeam", {})
        game.home_shots = home_stats.get("sog")
        game.away_shots = away_stats.get("sog")

        await session.commit()
        return game

    async def _upsert_game(
        self, session: AsyncSession, data: dict[str, Any]
    ) -> HockeyGame | None:
        ext_id = str(data.get("id", ""))
        if not ext_id:
            return None

        # Look up teams first to avoid adding a game with NULL team IDs
        home_abbrev = data.get("homeTeam", {}).get("abbrev", "")
        away_abbrev = data.get("awayTeam", {}).get("abbrev", "")

        home_result = await session.execute(
            select(HockeyTeam).where(HockeyTeam.external_id == home_abbrev)
        )
        away_result = await session.execute(
            select(HockeyTeam).where(HockeyTeam.external_id == away_abbrev)
        )
        home_team = home_result.scalar_one_or_none()
        away_team = away_result.scalar_one_or_none()

        if not home_team or not away_team:
            return None

        result = await session.execute(
            select(HockeyGame).where(HockeyGame.external_id == ext_id)
        )
        game = result.scalar_one_or_none()
        if game is None:
            game = HockeyGame(external_id=ext_id)
            session.add(game)

        game.home_team_id = home_team.id
        game.away_team_id = away_team.id

        start_time = data.get("startTimeUTC", "")
        if start_time:
            game.game_date = datetime.fromisoformat(
                start_time.replace("Z", "+00:00")
            )

        game.season = data.get("season")
        game.game_type = str(data.get("gameType", ""))

        state = data.get("gameState", "")
        if state in ("FINAL", "OFF"):
            game.status = "FINISHED"
            game.home_score = data.get("homeTeam", {}).get("score")
            game.away_score = data.get("awayTeam", {}).get("score")
            if game.home_score is not None and game.away_score is not None:
                game.result = "H" if game.home_score > game.away_score else "A"
        else:
            game.status = "SCHEDULED"

        return game

    async def collect_all(self, session: AsyncSession) -> None:
        """Collect teams and schedule."""
        await self.collect_teams(session)
        await self.collect_schedule(session)
