"""Tennis data collector using ESPN scoreboard endpoints."""

import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from puntboom.collectors.base import BaseCollector
from puntboom.models.tennis import TennisMatch, TennisPlayer

logger = logging.getLogger(__name__)

GRAND_SLAMS = {"Australian Open", "Roland Garros", "Wimbledon", "US Open"}


class TennisCollector(BaseCollector):
    BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/tennis"
    RATE_LIMIT = 20

    def __init__(self):
        super().__init__()

    async def collect_scoreboard(
        self,
        session: AsyncSession,
        tour: str = "atp",
        date: str | None = None,
    ) -> list[TennisMatch]:
        """Fetch tennis scoreboard. tour: 'atp' or 'wta'."""
        params: dict[str, Any] = {}
        if date:
            params["dates"] = date

        data = await self.get(f"/{tour}/scoreboard", params=params)
        events = data.get("events", [])
        matches = []

        for event in events:
            tournament_name = event.get("name", "")
            is_gs = any(gs in tournament_name for gs in GRAND_SLAMS)

            for comp in event.get("competitions", []):
                try:
                    match = await self._process_competition(
                        session, comp, tournament_name, is_gs
                    )
                    if match:
                        matches.append(match)
                except Exception:
                    logger.exception(f"Failed to process competition in {tournament_name}")

        await session.commit()
        logger.info(f"Collected {len(matches)} tennis matches ({tour})")
        return matches

    async def _process_competition(
        self,
        session: AsyncSession,
        comp: dict[str, Any],
        tournament: str,
        is_grand_slam: bool,
    ) -> TennisMatch | None:
        competitors = comp.get("competitors", [])
        if len(competitors) < 2:
            return None

        player1_data = competitors[0]
        player2_data = competitors[1]
        p1 = await self._upsert_player(session, player1_data)
        p2 = await self._upsert_player(session, player2_data)

        ext_id = str(comp.get("id", ""))
        if not ext_id:
            return None

        result = await session.execute(
            select(TennisMatch).where(TennisMatch.external_id == ext_id)
        )
        match = result.scalar_one_or_none()
        if match is None:
            match = TennisMatch(external_id=ext_id)
            session.add(match)

        match.player1_id = p1.id
        match.player2_id = p2.id
        match.tournament = tournament
        match.is_grand_slam = is_grand_slam

        date_str = comp.get("date", "")
        if date_str:
            match.match_date = datetime.fromisoformat(
                date_str.replace("Z", "+00:00")
            )

        status = comp.get("status", {})
        status_name = status.get("type", {}).get("name", "")
        if status_name == "STATUS_FINAL":
            match.status = "FINISHED"
            winner_id = comp.get("winner", {}).get("id")
            if winner_id:
                match.result = "1" if str(winner_id) == player1_data.get("id") else "2"

            # Parse set scores
            p1_sets = 0
            p2_sets = 0
            score_parts = []
            for s in comp.get("sets", []):
                s1 = s.get("team1Score", 0)
                s2 = s.get("team2Score", 0)
                if s1 > s2:
                    p1_sets += 1
                elif s2 > s1:
                    p2_sets += 1
                score_parts.append(f"{s1}-{s2}")
            match.player1_sets = p1_sets
            match.player2_sets = p2_sets
            match.score_detail = " ".join(score_parts)
        else:
            match.status = "SCHEDULED"

        # Try to get round info from notes
        for note in comp.get("notes", []):
            if "round" in note.get("type", "").lower():
                match.round = note.get("headline", "")

        return match

    async def _upsert_player(
        self, session: AsyncSession, data: dict[str, Any]
    ) -> TennisPlayer:
        athlete = data.get("athlete", data)
        ext_id = str(athlete.get("id", data.get("id", "")))

        result = await session.execute(
            select(TennisPlayer).where(TennisPlayer.external_id == ext_id)
        )
        player = result.scalar_one_or_none()
        if player is None:
            player = TennisPlayer(
                external_id=ext_id,
                name=athlete.get("displayName", athlete.get("name", "Unknown")),
                country=athlete.get("flag", {}).get("alt"),
            )
            session.add(player)
            await session.flush()

        rank = data.get("rank")
        if rank:
            player.ranking = int(rank)

        return player

    async def collect_all(self, session: AsyncSession) -> None:
        """Collect ATP and WTA scoreboards."""
        await self.collect_scoreboard(session, "atp")
        await self.collect_scoreboard(session, "wta")
