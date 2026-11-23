"""Odds data collector using the-odds-api.com v4."""

import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from puntboom.collectors.base import BaseCollector
from puntboom.config import settings
from puntboom.models.odds import OddsSnapshot

logger = logging.getLogger(__name__)

# Mapping from odds-api sport keys to our sport identifiers
SPORT_KEYS = {
    "soccer_epl": "football",
    "soccer_germany_bundesliga": "football",
    "soccer_italy_serie_a": "football",
    "soccer_spain_la_liga": "football",
    "soccer_france_ligue_one": "football",
    "soccer_uefa_champs_league": "football",
    "basketball_nba": "basketball",
    "icehockey_nhl": "hockey",
    "tennis_atp_french_open": "tennis",
    "tennis_atp_wimbledon": "tennis",
    "tennis_atp_us_open": "tennis",
    "tennis_atp_australian_open": "tennis",
}


class OddsCollector(BaseCollector):
    BASE_URL = "https://api.the-odds-api.com/v4"
    RATE_LIMIT = 30

    def __init__(self):
        super().__init__(api_key=settings.odds_api_key)

    async def collect_odds(
        self,
        session: AsyncSession,
        sport_key: str = "soccer_epl",
        markets: str = "h2h",
    ) -> list[OddsSnapshot]:
        """Fetch current odds for a sport."""
        params = {
            "apiKey": self.api_key,
            "regions": "eu",
            "markets": markets,
            "oddsFormat": "decimal",
        }
        data = await self.get(f"/sports/{sport_key}/odds", params=params)
        if not isinstance(data, list):
            data = []

        our_sport = SPORT_KEYS.get(sport_key, "unknown")
        snapshots = []
        now = datetime.now(timezone.utc)

        for event in data:
            event_id = event.get("id", "")
            for bookmaker in event.get("bookmakers", []):
                bk_name = bookmaker.get("title", "")
                for market in bookmaker.get("markets", []):
                    market_key = market.get("key", "")
                    outcomes = market.get("outcomes", [])
                    if len(outcomes) < 2:
                        continue

                    snapshot = OddsSnapshot(
                        sport=our_sport,
                        match_external_id=event_id,
                        bookmaker=bk_name,
                        market=market_key,
                        snapshot_time=now,
                        outcome1_label=outcomes[0].get("name", ""),
                        outcome1_odds=outcomes[0].get("price", 0.0),
                        outcome2_label=outcomes[1].get("name", ""),
                        outcome2_odds=outcomes[1].get("price", 0.0),
                    )
                    if len(outcomes) > 2:
                        snapshot.outcome3_label = outcomes[2].get("name", "")
                        snapshot.outcome3_odds = outcomes[2].get("price", 0.0)

                    session.add(snapshot)
                    snapshots.append(snapshot)

        await session.commit()
        logger.info(
            f"Collected {len(snapshots)} odds snapshots for {sport_key}"
        )
        return snapshots

    async def collect_all(self, session: AsyncSession) -> None:
        """Collect odds for all tracked sport keys."""
        for sport_key in SPORT_KEYS:
            try:
                await self.collect_odds(session, sport_key)
            except Exception:
                logger.exception(f"Failed to collect odds for {sport_key}")
