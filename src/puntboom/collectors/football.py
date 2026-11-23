"""Football data collector using football-data.org v4 API."""

import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from puntboom.collectors.base import BaseCollector
from puntboom.config import settings
from puntboom.models.football import FootballLeague, FootballMatch, FootballTeam

logger = logging.getLogger(__name__)

# Free tier competitions
COMPETITIONS = {
    "PL": "Premier League",
    "BL1": "Bundesliga",
    "SA": "Serie A",
    "PD": "La Liga",
    "FL1": "Ligue 1",
    "CL": "Champions League",
}


class FootballCollector(BaseCollector):
    BASE_URL = "https://api.football-data.org/v4"
    RATE_LIMIT = 10

    def __init__(self):
        super().__init__(api_key=settings.football_data_api_key)

    @property
    def headers(self) -> dict[str, str]:
        return {"X-Auth-Token": self.api_key, "Accept": "application/json"}

    async def collect_competitions(self, session: AsyncSession) -> list[FootballLeague]:
        """Fetch and store available competitions."""
        leagues = []
        for code, name in COMPETITIONS.items():
            try:
                data = await self.get(f"/competitions/{code}")
                league = await self._upsert_league(session, data)
                leagues.append(league)
            except Exception:
                logger.exception(f"Failed to fetch competition {code}")
        await session.commit()
        return leagues

    async def _upsert_league(
        self, session: AsyncSession, data: dict[str, Any]
    ) -> FootballLeague:
        ext_id = str(data["id"])
        result = await session.execute(
            select(FootballLeague).where(FootballLeague.external_id == ext_id)
        )
        league = result.scalar_one_or_none()
        if league is None:
            league = FootballLeague(
                external_id=ext_id,
                name=data["name"],
                code=data.get("code", ""),
                country=data.get("area", {}).get("name", ""),
            )
            session.add(league)
        season = data.get("currentSeason")
        if season:
            league.current_season = int(str(season.get("startDate", ""))[:4]) or None
        return league

    async def collect_matches(
        self, session: AsyncSession, competition: str = "PL", season: int | None = None
    ) -> list[FootballMatch]:
        """Fetch matches for a competition."""
        # Look up the league so we can link matches to it
        league_result = await session.execute(
            select(FootballLeague).where(FootballLeague.code == competition)
        )
        league = league_result.scalar_one_or_none()

        params: dict[str, Any] = {}
        if season:
            params["season"] = season
        data = await self.get(f"/competitions/{competition}/matches", params=params)
        matches_data = data.get("matches", [])
        matches = []

        for m in matches_data:
            try:
                home_team = await self._upsert_team(session, m["homeTeam"])
                away_team = await self._upsert_team(session, m["awayTeam"])

                match = await self._upsert_match(session, m, home_team, away_team)
                if league:
                    match.league_id = league.id
                matches.append(match)
            except Exception:
                logger.exception(f"Failed to process match {m.get('id')}")

        await session.commit()
        logger.info(f"Collected {len(matches)} matches for {competition}")
        return matches

    async def _upsert_team(
        self, session: AsyncSession, data: dict[str, Any]
    ) -> FootballTeam:
        ext_id = str(data["id"])
        result = await session.execute(
            select(FootballTeam).where(FootballTeam.external_id == ext_id)
        )
        team = result.scalar_one_or_none()
        if team is None:
            team = FootballTeam(
                external_id=ext_id,
                name=data.get("name", "Unknown"),
                short_name=data.get("shortName"),
                crest_url=data.get("crest"),
            )
            session.add(team)
            await session.flush()
        return team

    async def _upsert_match(
        self,
        session: AsyncSession,
        data: dict[str, Any],
        home_team: FootballTeam,
        away_team: FootballTeam,
    ) -> FootballMatch:
        ext_id = str(data["id"])
        result = await session.execute(
            select(FootballMatch).where(FootballMatch.external_id == ext_id)
        )
        match = result.scalar_one_or_none()
        if match is None:
            match = FootballMatch(external_id=ext_id)
            session.add(match)

        match.home_team_id = home_team.id
        match.away_team_id = away_team.id
        match.match_date = datetime.fromisoformat(
            data["utcDate"].replace("Z", "+00:00")
        )
        match.status = data.get("status", "SCHEDULED")
        match.matchday = data.get("matchday")
        match.season = int(str(data.get("season", {}).get("startDate", ""))[:4]) or None

        score = data.get("score", {})
        ft = score.get("fullTime", {})
        ht = score.get("halfTime", {})
        match.home_score = ft.get("home")
        match.away_score = ft.get("away")
        match.home_ht_score = ht.get("home")
        match.away_ht_score = ht.get("away")

        if match.home_score is not None and match.away_score is not None:
            if match.home_score > match.away_score:
                match.result = "H"
            elif match.home_score < match.away_score:
                match.result = "A"
            else:
                match.result = "D"

        return match

    async def collect_all(self, session: AsyncSession) -> None:
        """Collect data for all free tier competitions."""
        await self.collect_competitions(session)
        for code in COMPETITIONS:
            try:
                await self.collect_matches(session, code)
            except Exception:
                logger.exception(f"Failed to collect matches for {code}")
