"""European ice hockey data collector using IceHockeyApi (RapidAPI).

Covers KHL, SHL, DEL, Liiga, National League A, and Czech Extraliga.
"""

import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from puntboom.collectors.base import BaseCollector
from puntboom.config import settings
from puntboom.models.hockey import HockeyGame, HockeyTeam

logger = logging.getLogger(__name__)

# Category IDs on IceHockeyApi
EURO_LEAGUES = {
    101: "KHL",
    39: "SHL",
    41: "DEL",
    40: "Liiga",
    54: "National League A",
    42: "Extraliga",
}

# Only keep events whose tournament name matches a target league.
# Each category also contains lower divisions (VHL, MHL, DEL 2, etc.)
# that we want to skip.
TARGET_TOURNAMENT_NAMES = {
    "KHL", "SHL", "DEL", "Liiga", "National League A", "Extraliga",
}


class EuroHockeyCollector(BaseCollector):
    BASE_URL = "https://icehockeyapi.p.rapidapi.com"
    RATE_LIMIT = 20  # conservative for free tier (50/day)

    def __init__(self):
        super().__init__(api_key=settings.rapidapi_key)

    @property
    def headers(self) -> dict[str, str]:
        return {
            "Accept": "application/json",
            "x-rapidapi-host": "icehockeyapi.p.rapidapi.com",
            "x-rapidapi-key": self.api_key,
        }

    async def collect_events(
        self, session: AsyncSession, date: datetime | None = None
    ) -> list[HockeyGame]:
        """Fetch all European hockey events for a given date."""
        if date is None:
            date = datetime.now(timezone.utc)

        day = date.day
        month = date.month
        year = date.year

        all_games: list[HockeyGame] = []

        for category_id, league_name in EURO_LEAGUES.items():
            path = (
                f"/api/ice-hockey/category/{category_id}"
                f"/events/{day}/{month}/{year}"
            )
            try:
                data = await self.get(path)
            except Exception:
                logger.exception(
                    f"Failed to fetch {league_name} (cat={category_id}) "
                    f"for {year}-{month:02d}-{day:02d}"
                )
                continue

            events = data.get("events", [])
            logger.info(
                f"Fetched {len(events)} events for {league_name} "
                f"on {year}-{month:02d}-{day:02d}"
            )

            for event in events:
                # Filter: only keep events from the target top-tier league
                tournament = event.get("tournament", {})
                unique_tournament = tournament.get("uniqueTournament", {})
                tourney_name = unique_tournament.get("name", "")
                if tourney_name not in TARGET_TOURNAMENT_NAMES:
                    continue

                try:
                    game = await self._upsert_game(session, event, league_name)
                    if game:
                        all_games.append(game)
                except Exception:
                    logger.exception(
                        f"Failed to process euro hockey event {event.get('id')}"
                    )

        await session.commit()
        logger.info(f"Collected {len(all_games)} European hockey games")
        return all_games

    async def _upsert_team(
        self,
        session: AsyncSession,
        team_data: dict[str, Any],
        league_name: str,
    ) -> HockeyTeam | None:
        """Create or retrieve a European hockey team.

        Uses ``eu_{api_team_id}`` as external_id to avoid collisions with
        NHL abbreviation-based IDs.
        """
        api_id = team_data.get("id")
        if not api_id:
            return None

        ext_id = f"eu_{api_id}"
        result = await session.execute(
            select(HockeyTeam).where(HockeyTeam.external_id == ext_id)
        )
        team = result.scalar_one_or_none()

        if team is None:
            name = team_data.get("name", f"Team {api_id}")
            short = team_data.get("shortName", name[:10])
            team = HockeyTeam(
                external_id=ext_id,
                name=name,
                abbreviation=short,
                conference=league_name,
                division=None,
            )
            session.add(team)
            await session.flush()
            logger.debug(f"Created euro hockey team: {name} ({ext_id})")

        return team

    async def _upsert_game(
        self,
        session: AsyncSession,
        event: dict[str, Any],
        league_name: str,
    ) -> HockeyGame | None:
        """Create or update a European hockey game from an API event."""
        api_id = event.get("id")
        if not api_id:
            return None

        ext_id = f"eu_{api_id}"

        # Upsert teams first
        home_data = event.get("homeTeam", {})
        away_data = event.get("awayTeam", {})
        home_team = await self._upsert_team(session, home_data, league_name)
        away_team = await self._upsert_team(session, away_data, league_name)
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

        # Parse game date from Unix timestamp
        ts = event.get("startTimestamp")
        if ts:
            game.game_date = datetime.fromtimestamp(ts, tz=timezone.utc)

        # Store league context in season field
        tournament = event.get("tournament", {})
        unique_tournament = tournament.get("uniqueTournament", {})
        game.season = unique_tournament.get("name", league_name)
        game.game_type = "euro"

        # Map status
        status_info = event.get("status", {})
        status_type = status_info.get("type", "")
        if status_type == "finished":
            game.status = "FINISHED"
        elif status_type == "notstarted":
            game.status = "SCHEDULED"
        elif status_type == "inprogress":
            game.status = "LIVE"
        else:
            game.status = status_type.upper() if status_type else "SCHEDULED"

        # Scores
        home_score_data = event.get("homeScore", {})
        away_score_data = event.get("awayScore", {})

        if game.status == "FINISHED":
            game.home_score = home_score_data.get("current")
            game.away_score = away_score_data.get("current")

            # Period scores
            game.home_p1 = home_score_data.get("period1")
            game.home_p2 = home_score_data.get("period2")
            game.home_p3 = home_score_data.get("period3")
            game.away_p1 = away_score_data.get("period1")
            game.away_p2 = away_score_data.get("period2")
            game.away_p3 = away_score_data.get("period3")

            # Result from winnerCode: 1=Home, 2=Away
            winner = event.get("winnerCode")
            if winner == 1:
                game.result = "H"
            elif winner == 2:
                game.result = "A"
            elif game.home_score is not None and game.away_score is not None:
                game.result = "H" if game.home_score > game.away_score else "A"

        return game

    async def collect_all(self, session: AsyncSession) -> None:
        """Collect today's European hockey events."""
        await self.collect_events(session)
