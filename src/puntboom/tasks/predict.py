"""Scheduled prediction generation for upcoming matches."""

import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from puntboom.bankroll.kelly import fractional_kelly, kelly_stake_amount
from puntboom.bankroll.value import expected_value, is_value_bet
from puntboom.config import settings
from puntboom.database import async_session
from puntboom.ml.predictor import SportPredictor, TotalsClassifier, TotalsPredictor
from puntboom.models.football import FootballLeague, FootballMatch, FootballTeam
from puntboom.models.basketball import BasketballGame, BasketballTeam
from puntboom.models.hockey import HockeyGame, HockeyTeam
from puntboom.models.tennis import TennisMatch, TennisPlayer
from puntboom.models.prediction import Prediction
from puntboom.features.football import build_football_features
from puntboom.features.basketball import build_basketball_features
from puntboom.features.hockey import build_hockey_features
from puntboom.features.tennis import build_tennis_features

logger = logging.getLogger(__name__)


async def generate_predictions() -> None:
    """Generate predictions for all upcoming matches."""
    logger.info("Starting prediction generation")
    async with async_session() as session:
        await _predict_football(session)
        await _predict_basketball(session)
        await _predict_hockey(session)
        await _predict_tennis(session)
    logger.info("Finished prediction generation")


async def _predict_football(session: AsyncSession) -> None:
    predictor = SportPredictor("football")
    if not predictor.load():
        logger.warning("No football model available")
        return

    totals_predictor = TotalsPredictor("football")
    has_totals = totals_predictor.load()
    if not has_totals:
        logger.warning("No football totals model — O/U predictions will be skipped")

    totals_classifier = TotalsClassifier("football")
    has_classifier = totals_classifier.load()
    if not has_classifier:
        logger.info("No football O/U classifier — falling back to Poisson CDF")

    now = datetime.now(timezone.utc)
    lookahead = now + timedelta(days=14)
    result = await session.execute(
        select(FootballMatch)
        .where(
            and_(
                FootballMatch.status.in_(["SCHEDULED", "TIMED"]),
                FootballMatch.match_date >= now,
                FootballMatch.match_date <= lookahead,
            )
        )
        .order_by(FootballMatch.match_date)
        .limit(500)
    )
    matches = result.scalars().all()

    for match in matches:
        # Check if prediction already exists
        existing = await session.execute(
            select(Prediction).where(
                Prediction.match_external_id == match.external_id,
                Prediction.sport == "football",
            )
        )
        if existing.scalar_one_or_none():
            continue

        home_result = await session.execute(
            select(FootballTeam).where(FootballTeam.id == match.home_team_id)
        )
        away_result = await session.execute(
            select(FootballTeam).where(FootballTeam.id == match.away_team_id)
        )
        home_team = home_result.scalar_one_or_none()
        away_team = away_result.scalar_one_or_none()
        if not home_team or not away_team:
            continue

        try:
            features = await build_football_features(
                session, home_team, away_team, match.match_date
            )
            probs = predictor.predict(features)
            if not probs:
                continue

            # Totals: use regression for xGoals, classifier for O/U probs
            totals = None
            if has_totals:
                totals = totals_predictor.predict(features)
            if has_classifier and totals:
                cls_pred = totals_classifier.predict(features)
                if cls_pred:
                    totals["prob_over"] = cls_pred["prob_over"]
                    totals["prob_under"] = cls_pred["prob_under"]
                    totals["prediction"] = cls_pred["prediction"]

            # Look up league name
            league_name = None
            if match.league_id:
                league_res = await session.execute(
                    select(FootballLeague).where(FootballLeague.id == match.league_id)
                )
                league_obj = league_res.scalar_one_or_none()
                if league_obj:
                    league_name = league_obj.name

            _save_prediction(
                session, "football", match.external_id, match.match_date,
                home_team.name, away_team.name, probs,
                league=league_name, has_draw=True, totals=totals,
                home_avg_scored=features.get("home_goals_avg"),
                away_avg_scored=features.get("away_goals_avg"),
                home_avg_conceded=features.get("home_conceded_avg"),
                away_avg_conceded=features.get("away_conceded_avg"),
            )
        except Exception:
            logger.exception(f"Failed to predict football match {match.external_id}")

    await session.commit()


async def _predict_basketball(session: AsyncSession) -> None:
    predictor = SportPredictor("basketball")
    if not predictor.load():
        logger.warning("No basketball model available")
        return

    now = datetime.now(timezone.utc)
    result = await session.execute(
        select(BasketballGame)
        .where(
            and_(
                BasketballGame.status == "SCHEDULED",
                BasketballGame.game_date >= now,
            )
        )
        .limit(100)
    )
    games = result.scalars().all()

    for game in games:
        existing = await session.execute(
            select(Prediction).where(
                Prediction.match_external_id == game.external_id,
                Prediction.sport == "basketball",
            )
        )
        if existing.scalar_one_or_none():
            continue

        home_result = await session.execute(
            select(BasketballTeam).where(BasketballTeam.id == game.home_team_id)
        )
        away_result = await session.execute(
            select(BasketballTeam).where(BasketballTeam.id == game.away_team_id)
        )
        home_team = home_result.scalar_one_or_none()
        away_team = away_result.scalar_one_or_none()
        if not home_team or not away_team:
            continue

        try:
            features = await build_basketball_features(
                session, home_team, away_team, game.game_date
            )
            probs = predictor.predict(features)
            if not probs:
                continue

            _save_prediction(
                session, "basketball", game.external_id, game.game_date,
                home_team.name, away_team.name, probs,
            )
        except Exception:
            logger.exception(f"Failed to predict basketball game {game.external_id}")

    await session.commit()


async def _predict_hockey(session: AsyncSession) -> None:
    predictor = SportPredictor("hockey")
    if not predictor.load():
        logger.warning("No hockey model available")
        return

    totals_predictor = TotalsPredictor("hockey")
    has_totals = totals_predictor.load()
    if not has_totals:
        logger.warning("No hockey totals model — O/U predictions will be skipped")

    totals_classifier = TotalsClassifier("hockey")
    has_classifier = totals_classifier.load()
    if not has_classifier:
        logger.info("No hockey O/U classifier — falling back to Poisson CDF")

    now = datetime.now(timezone.utc)
    lookahead = now + timedelta(days=14)
    result = await session.execute(
        select(HockeyGame)
        .where(
            and_(
                HockeyGame.status == "SCHEDULED",
                HockeyGame.game_date >= now,
                HockeyGame.game_date <= lookahead,
            )
        )
        .order_by(HockeyGame.game_date)
        .limit(500)
    )
    games = result.scalars().all()

    for game in games:
        existing = await session.execute(
            select(Prediction).where(
                Prediction.match_external_id == game.external_id,
                Prediction.sport == "hockey",
            )
        )
        if existing.scalar_one_or_none():
            continue

        home_result = await session.execute(
            select(HockeyTeam).where(HockeyTeam.id == game.home_team_id)
        )
        away_result = await session.execute(
            select(HockeyTeam).where(HockeyTeam.id == game.away_team_id)
        )
        home_team = home_result.scalar_one_or_none()
        away_team = away_result.scalar_one_or_none()
        if not home_team or not away_team:
            continue

        try:
            features = await build_hockey_features(
                session, home_team, away_team, game.game_date
            )
            probs = predictor.predict(features)
            if not probs:
                continue

            # Hockey totals with 5.5 line
            totals = None
            if has_totals:
                totals = totals_predictor.predict(features, line=5.5)
            if has_classifier and totals:
                cls_pred = totals_classifier.predict(features)
                if cls_pred:
                    totals["prob_over"] = cls_pred["prob_over"]
                    totals["prob_under"] = cls_pred["prob_under"]
                    totals["prediction"] = cls_pred["prediction"]

            # Derive league name: euro games store tournament name in season,
            # NHL games store year like "20262027"
            if game.external_id.startswith("eu_"):
                league_name = game.season
            else:
                league_name = "NHL"

            _save_prediction(
                session, "hockey", game.external_id, game.game_date,
                home_team.name, away_team.name, probs,
                league=league_name, totals=totals, total_line_override=5.5,
                home_avg_scored=features.get("home_goals_per_game"),
                away_avg_scored=features.get("away_goals_per_game"),
                home_avg_conceded=features.get("home_goals_against_avg"),
                away_avg_conceded=features.get("away_goals_against_avg"),
            )
        except Exception:
            logger.exception(f"Failed to predict hockey game {game.external_id}")

    await session.commit()


async def _predict_tennis(session: AsyncSession) -> None:
    predictor = SportPredictor("tennis")
    if not predictor.load():
        logger.warning("No tennis model available")
        return

    now = datetime.now(timezone.utc)
    result = await session.execute(
        select(TennisMatch)
        .where(
            and_(
                TennisMatch.status == "SCHEDULED",
                TennisMatch.match_date >= now,
            )
        )
        .limit(100)
    )
    matches = result.scalars().all()

    for match in matches:
        existing = await session.execute(
            select(Prediction).where(
                Prediction.match_external_id == match.external_id,
                Prediction.sport == "tennis",
            )
        )
        if existing.scalar_one_or_none():
            continue

        p1_result = await session.execute(
            select(TennisPlayer).where(TennisPlayer.id == match.player1_id)
        )
        p2_result = await session.execute(
            select(TennisPlayer).where(TennisPlayer.id == match.player2_id)
        )
        player1 = p1_result.scalar_one_or_none()
        player2 = p2_result.scalar_one_or_none()
        if not player1 or not player2:
            continue

        try:
            features = await build_tennis_features(
                session, player1, player2, match.match_date,
                surface=match.surface, round_name=match.round,
                is_grand_slam=match.is_grand_slam or False,
            )
            probs = predictor.predict(features)
            if not probs:
                continue

            # Tennis uses "1"/"2" labels mapped to home/away
            mapped_probs = {}
            mapped_probs["H"] = probs.get("1", 0.0)
            mapped_probs["A"] = probs.get("2", 0.0)

            _save_prediction(
                session, "tennis", match.external_id, match.match_date,
                player1.name, player2.name, mapped_probs,
                league=match.tournament,
            )
        except Exception:
            logger.exception(f"Failed to predict tennis match {match.external_id}")

    await session.commit()


def _save_prediction(
    session: AsyncSession,
    sport: str,
    match_external_id: str,
    match_date: datetime,
    home_name: str,
    away_name: str,
    probs: dict[str, float],
    league: str | None = None,
    has_draw: bool = False,
    totals: dict[str, float] | None = None,
    total_line_override: float | None = None,
    home_avg_scored: float | None = None,
    away_avg_scored: float | None = None,
    home_avg_conceded: float | None = None,
    away_avg_conceded: float | None = None,
) -> None:
    """Create and save a Prediction record."""
    prob_home = probs.get("H", 0.0)
    prob_draw = probs.get("D") if has_draw else None
    prob_away = probs.get("A", 0.0)

    # Determine predicted outcome
    if has_draw:
        outcomes = {"H": prob_home, "D": prob_draw or 0.0, "A": prob_away}
    else:
        outcomes = {"H": prob_home, "A": prob_away}
    predicted = max(outcomes, key=outcomes.get)
    confidence = outcomes[predicted]

    # Placeholder odds (would come from odds snapshots in production)
    # For now, use implied probability inverse as rough odds
    odds_home = 1.0 / prob_home if prob_home > 0.05 else None
    odds_draw = 1.0 / prob_draw if prob_draw and prob_draw > 0.05 else None
    odds_away = 1.0 / prob_away if prob_away > 0.05 else None

    # Value bet check
    pred_odds = odds_home if predicted == "H" else (odds_draw if predicted == "D" else odds_away)
    ev = expected_value(confidence, pred_odds) if pred_odds else 0.0
    is_vb = is_value_bet(confidence, pred_odds) if pred_odds else False
    kelly = kelly_stake_amount(confidence, pred_odds, settings.default_bankroll) if pred_odds and pred_odds > 1 else 0.0

    # Totals / O/U data
    expected_goals = None
    total_line = None
    prob_over = None
    prob_under = None
    predicted_total = None
    if totals:
        expected_goals = totals.get("expected_total")
        total_line = total_line_override if total_line_override is not None else 2.5
        prob_over = totals.get("prob_over")
        prob_under = totals.get("prob_under")
        # Use classifier's threshold-aware prediction if available,
        # otherwise fall back to 0.5
        predicted_total = totals.get(
            "prediction", "Over" if (prob_over or 0) >= 0.5 else "Under"
        )

    prediction = Prediction(
        sport=sport,
        match_external_id=match_external_id,
        match_date=match_date,
        home_team=home_name,
        away_team=away_name,
        league=league,
        prob_home=prob_home,
        prob_draw=prob_draw,
        prob_away=prob_away,
        predicted_outcome=predicted,
        confidence=confidence,
        odds_home=odds_home,
        odds_draw=odds_draw,
        odds_away=odds_away,
        is_value_bet=is_vb,
        expected_value=ev,
        kelly_stake=kelly,
        home_avg_scored=home_avg_scored,
        away_avg_scored=away_avg_scored,
        home_avg_conceded=home_avg_conceded,
        away_avg_conceded=away_avg_conceded,
        expected_goals=expected_goals,
        total_line=total_line,
        prob_over=prob_over,
        prob_under=prob_under,
        predicted_total=predicted_total,
    )
    session.add(prediction)
