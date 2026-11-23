"""Scheduled model retraining."""

import logging

import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from puntboom.database import async_session
from puntboom.features.football import build_football_features, match_results_for_team
from puntboom.features.basketball import build_basketball_features
from puntboom.features.hockey import build_hockey_features
from puntboom.features.tennis import build_tennis_features
from puntboom.ml.trainer import train_sport_model, train_totals_model
from puntboom.ml.registry import ModelRegistry
from puntboom.models.football import FootballMatch, FootballTeam
from puntboom.models.basketball import BasketballGame, BasketballTeam
from puntboom.models.hockey import HockeyGame, HockeyTeam
from puntboom.models.tennis import TennisMatch, TennisPlayer

logger = logging.getLogger(__name__)


async def retrain_all_models() -> dict:
    """Retrain models for all sports."""
    logger.info("Starting model retraining")
    results = {}
    registry = ModelRegistry()

    async with async_session() as session:
        for sport, func in [
            ("football", _build_football_dataset),
            ("basketball", _build_basketball_dataset),
            ("hockey", _build_hockey_dataset),
            ("tennis", _build_tennis_dataset),
        ]:
            try:
                features_df, labels = await func(session)
                if features_df.empty:
                    logger.warning(f"No training data for {sport}")
                    results[sport] = {"error": "no_data"}
                    continue

                result = train_sport_model(sport, features_df, labels)
                results[sport] = result

                if "version" in result:
                    registry.register(
                        sport=sport,
                        version=result["version"],
                        model_type=result.get("best_model", "unknown"),
                        metrics=result.get("metrics", {}),
                        sample_count=result.get("samples", 0),
                    )
            except Exception:
                logger.exception(f"Failed to retrain {sport}")
                results[sport] = {"error": "training_failed"}

        # Train totals (regression) models
        for sport, func in [
            ("football", _build_football_totals_dataset),
            ("hockey", _build_hockey_totals_dataset),
        ]:
            try:
                features_df, labels = await func(session)
                if features_df.empty:
                    logger.warning(f"No totals training data for {sport}")
                    results[f"{sport}_totals"] = {"error": "no_data"}
                    continue

                result = train_totals_model(sport, features_df, labels)
                results[f"{sport}_totals"] = result

                if "version" in result:
                    registry.register(
                        sport=f"{sport}_totals",
                        version=result["version"],
                        model_type=result.get("best_model", "unknown"),
                        metrics=result.get("metrics", {}),
                        sample_count=result.get("samples", 0),
                    )
            except Exception:
                logger.exception(f"Failed to retrain {sport} totals")
                results[f"{sport}_totals"] = {"error": "training_failed"}

    logger.info("Finished model retraining")
    return results


async def _build_football_dataset(
    session: AsyncSession,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build training dataset for football."""
    result = await session.execute(
        select(FootballMatch)
        .where(FootballMatch.status == "FINISHED", FootballMatch.result.isnot(None))
        .order_by(FootballMatch.match_date)
        .limit(5000)
    )
    matches = result.scalars().all()

    rows = []
    labels = []
    for match in matches:
        home_res = await session.execute(
            select(FootballTeam).where(FootballTeam.id == match.home_team_id)
        )
        away_res = await session.execute(
            select(FootballTeam).where(FootballTeam.id == match.away_team_id)
        )
        home_team = home_res.scalar_one_or_none()
        away_team = away_res.scalar_one_or_none()
        if not home_team or not away_team:
            continue

        try:
            features = await build_football_features(
                session, home_team, away_team, match.match_date
            )
            rows.append(features)
            labels.append(match.result)
        except Exception:
            continue

    return pd.DataFrame(rows), pd.Series(labels)


async def _build_basketball_dataset(
    session: AsyncSession,
) -> tuple[pd.DataFrame, pd.Series]:
    result = await session.execute(
        select(BasketballGame)
        .where(BasketballGame.status == "FINISHED", BasketballGame.result.isnot(None))
        .order_by(BasketballGame.game_date)
        .limit(5000)
    )
    games = result.scalars().all()

    rows = []
    labels = []
    for game in games:
        home_res = await session.execute(
            select(BasketballTeam).where(BasketballTeam.id == game.home_team_id)
        )
        away_res = await session.execute(
            select(BasketballTeam).where(BasketballTeam.id == game.away_team_id)
        )
        home_team = home_res.scalar_one_or_none()
        away_team = away_res.scalar_one_or_none()
        if not home_team or not away_team:
            continue

        try:
            features = await build_basketball_features(
                session, home_team, away_team, game.game_date
            )
            rows.append(features)
            labels.append(game.result)
        except Exception:
            continue

    return pd.DataFrame(rows), pd.Series(labels)


async def _build_hockey_dataset(
    session: AsyncSession,
) -> tuple[pd.DataFrame, pd.Series]:
    result = await session.execute(
        select(HockeyGame)
        .where(HockeyGame.status == "FINISHED", HockeyGame.result.isnot(None))
        .order_by(HockeyGame.game_date)
        .limit(5000)
    )
    games = result.scalars().all()

    rows = []
    labels = []
    for game in games:
        home_res = await session.execute(
            select(HockeyTeam).where(HockeyTeam.id == game.home_team_id)
        )
        away_res = await session.execute(
            select(HockeyTeam).where(HockeyTeam.id == game.away_team_id)
        )
        home_team = home_res.scalar_one_or_none()
        away_team = away_res.scalar_one_or_none()
        if not home_team or not away_team:
            continue

        try:
            features = await build_hockey_features(
                session, home_team, away_team, game.game_date
            )
            rows.append(features)
            labels.append(game.result)
        except Exception:
            continue

    return pd.DataFrame(rows), pd.Series(labels)


async def _build_tennis_dataset(
    session: AsyncSession,
) -> tuple[pd.DataFrame, pd.Series]:
    result = await session.execute(
        select(TennisMatch)
        .where(TennisMatch.status == "FINISHED", TennisMatch.result.isnot(None))
        .order_by(TennisMatch.match_date)
        .limit(5000)
    )
    matches = result.scalars().all()

    rows = []
    labels = []
    for match in matches:
        p1_res = await session.execute(
            select(TennisPlayer).where(TennisPlayer.id == match.player1_id)
        )
        p2_res = await session.execute(
            select(TennisPlayer).where(TennisPlayer.id == match.player2_id)
        )
        player1 = p1_res.scalar_one_or_none()
        player2 = p2_res.scalar_one_or_none()
        if not player1 or not player2:
            continue

        try:
            features = await build_tennis_features(
                session, player1, player2, match.match_date,
                surface=match.surface, round_name=match.round,
                is_grand_slam=match.is_grand_slam or False,
            )
            rows.append(features)
            labels.append(match.result)
        except Exception:
            continue

    return pd.DataFrame(rows), pd.Series(labels)


async def _build_football_totals_dataset(
    session: AsyncSession,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build training dataset for football totals (goals regression).

    Labels are total goals (home_score + away_score).
    """
    result = await session.execute(
        select(FootballMatch)
        .where(
            FootballMatch.status == "FINISHED",
            FootballMatch.home_score.isnot(None),
            FootballMatch.away_score.isnot(None),
        )
        .order_by(FootballMatch.match_date)
        .limit(5000)
    )
    matches = result.scalars().all()

    rows = []
    labels = []
    for match in matches:
        home_res = await session.execute(
            select(FootballTeam).where(FootballTeam.id == match.home_team_id)
        )
        away_res = await session.execute(
            select(FootballTeam).where(FootballTeam.id == match.away_team_id)
        )
        home_team = home_res.scalar_one_or_none()
        away_team = away_res.scalar_one_or_none()
        if not home_team or not away_team:
            continue

        try:
            features = await build_football_features(
                session, home_team, away_team, match.match_date
            )
            rows.append(features)
            labels.append(match.home_score + match.away_score)
        except Exception:
            continue

    return pd.DataFrame(rows), pd.Series(labels, dtype=float)


async def _build_hockey_totals_dataset(
    session: AsyncSession,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build training dataset for hockey totals (goals regression).

    Labels are total goals (home_score + away_score).
    """
    result = await session.execute(
        select(HockeyGame)
        .where(
            HockeyGame.status == "FINISHED",
            HockeyGame.home_score.isnot(None),
            HockeyGame.away_score.isnot(None),
        )
        .order_by(HockeyGame.game_date)
        .limit(5000)
    )
    games = result.scalars().all()

    rows = []
    labels = []
    for game in games:
        home_res = await session.execute(
            select(HockeyTeam).where(HockeyTeam.id == game.home_team_id)
        )
        away_res = await session.execute(
            select(HockeyTeam).where(HockeyTeam.id == game.away_team_id)
        )
        home_team = home_res.scalar_one_or_none()
        away_team = away_res.scalar_one_or_none()
        if not home_team or not away_team:
            continue

        try:
            features = await build_hockey_features(
                session, home_team, away_team, game.game_date
            )
            rows.append(features)
            labels.append(game.home_score + game.away_score)
        except Exception:
            continue

    return pd.DataFrame(rows), pd.Series(labels, dtype=float)
