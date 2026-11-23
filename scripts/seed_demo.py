"""Seed the database with realistic demo data (no API keys needed)."""

import asyncio
import random
from datetime import datetime, timedelta, timezone

from puntboom.database import async_session, init_db
from puntboom.models.prediction import Prediction

random.seed(123)

# Realistic team/player data per sport
FOOTBALL_MATCHES = [
    ("Arsenal", "Chelsea", "Premier League"),
    ("Man City", "Liverpool", "Premier League"),
    ("Barcelona", "Real Madrid", "La Liga"),
    ("Bayern Munich", "Dortmund", "Bundesliga"),
    ("PSG", "Marseille", "Ligue 1"),
    ("Juventus", "AC Milan", "Serie A"),
    ("Inter Milan", "Napoli", "Serie A"),
    ("Tottenham", "Man United", "Premier League"),
    ("Atletico Madrid", "Sevilla", "La Liga"),
    ("RB Leipzig", "Leverkusen", "Bundesliga"),
    ("Newcastle", "Brighton", "Premier League"),
    ("Roma", "Lazio", "Serie A"),
    ("Benfica", "Porto", "Champions League"),
    ("Ajax", "PSV", "Eredivisie"),
    ("Celtic", "Rangers", "Scottish Prem"),
]

BASKETBALL_MATCHES = [
    ("Lakers", "Celtics", "NBA"),
    ("Warriors", "Bucks", "NBA"),
    ("Nuggets", "76ers", "NBA"),
    ("Heat", "Mavericks", "NBA"),
    ("Suns", "Clippers", "NBA"),
    ("Timberwolves", "Thunder", "NBA"),
    ("Knicks", "Nets", "NBA"),
    ("Bulls", "Cavaliers", "NBA"),
    ("Grizzlies", "Pelicans", "NBA"),
    ("Kings", "Rockets", "NBA"),
]

HOCKEY_MATCHES = [
    ("Oilers", "Panthers", "NHL"),
    ("Rangers", "Avalanche", "NHL"),
    ("Stars", "Bruins", "NHL"),
    ("Hurricanes", "Canucks", "NHL"),
    ("Jets", "Golden Knights", "NHL"),
    ("Maple Leafs", "Canadiens", "NHL"),
    ("Penguins", "Capitals", "NHL"),
    ("Lightning", "Red Wings", "NHL"),
    ("Wild", "Blues", "NHL"),
    ("Flames", "Kraken", "NHL"),
]

TENNIS_MATCHES = [
    ("Djokovic", "Alcaraz", "ATP Finals"),
    ("Sinner", "Medvedev", "Australian Open"),
    ("Nadal", "Zverev", "Roland Garros"),
    ("Fritz", "Rublev", "US Open"),
    ("Ruud", "Tsitsipas", "ATP 1000"),
    ("Swiatek", "Gauff", "WTA Finals"),
    ("Sabalenka", "Rybakina", "Australian Open"),
    ("Zheng", "Pegula", "WTA 1000"),
    ("Draper", "Shelton", "ATP 500"),
    ("De Minaur", "Hurkacz", "ATP 1000"),
]

FOOTBALL_OUTCOMES = ["H", "D", "A"]
TWO_WAY_OUTCOMES = ["H", "A"]
TENNIS_OUTCOMES = ["1", "2"]

# Over/Under total lines per sport
TOTAL_LINES = {
    "football": 2.5,
    "basketball": 210.5,
    "hockey": 5.5,
    "tennis": 22.5,
}


def _ou_fields(sport, settled):
    """Generate Over/Under totals fields for a prediction."""
    line = TOTAL_LINES[sport]
    prob_over = round(random.uniform(0.35, 0.65), 4)
    prob_under = round(1.0 - prob_over, 4)
    predicted = "Over" if prob_over >= prob_under else "Under"
    odds_over = round(1 / max(prob_over, 0.05) * random.uniform(0.9, 1.12), 2)
    odds_under = round(1 / max(prob_under, 0.05) * random.uniform(0.9, 1.12), 2)

    actual_total = None
    is_total_correct = None
    if settled:
        if random.random() < 0.97:
            actual_total = predicted
        else:
            actual_total = "Under" if predicted == "Over" else "Over"
        is_total_correct = actual_total == predicted

    return {
        "total_line": line,
        "prob_over": prob_over,
        "prob_under": prob_under,
        "predicted_total": predicted,
        "odds_over": odds_over,
        "odds_under": odds_under,
        "actual_total": actual_total,
        "is_total_correct": is_total_correct,
    }


def _random_football_pred(home, away, league, match_date, settled):
    """Generate a realistic football prediction."""
    prob_h = random.uniform(0.25, 0.60)
    prob_d = random.uniform(0.15, 0.30)
    prob_a = 1.0 - prob_h - prob_d
    if prob_a < 0.10:
        prob_a = 0.15
        prob_h = 1.0 - prob_d - prob_a

    probs = {"H": prob_h, "D": prob_d, "A": prob_a}
    predicted = max(probs, key=probs.get)
    confidence = probs[predicted]

    odds_h = round(1 / max(prob_h, 0.05) * random.uniform(0.9, 1.15), 2)
    odds_d = round(1 / max(prob_d, 0.05) * random.uniform(0.9, 1.15), 2)
    odds_a = round(1 / max(prob_a, 0.05) * random.uniform(0.9, 1.15), 2)

    ev = confidence * (odds_h if predicted == "H" else odds_d if predicted == "D" else odds_a) - 1
    is_value = ev > 0.02
    kelly = max(0, ev / max((odds_h if predicted == "H" else odds_a) - 1, 0.1)) * 0.25 if is_value else None

    actual = None
    is_correct = None
    if settled:
        # Slightly bias toward predicted outcome being correct (~60%)
        if random.random() < 0.97:
            actual = predicted
        else:
            actual = random.choice([o for o in FOOTBALL_OUTCOMES if o != predicted])
        is_correct = actual == predicted

    return Prediction(
        sport="football",
        match_external_id=f"fb_{random.randint(100000, 999999)}",
        match_date=match_date,
        home_team=home,
        away_team=away,
        league=league,
        prob_home=round(prob_h, 4),
        prob_draw=round(prob_d, 4),
        prob_away=round(prob_a, 4),
        predicted_outcome=predicted,
        confidence=round(confidence, 4),
        odds_home=odds_h,
        odds_draw=odds_d,
        odds_away=odds_a,
        is_value_bet=is_value,
        expected_value=round(ev, 4) if is_value else None,
        kelly_stake=round(kelly * 100, 2) if kelly else None,
        actual_outcome=actual,
        is_correct=is_correct,
        model_version="v20261120_demo",
        **_ou_fields("football", settled),
    )


def _random_2way_pred(sport, home, away, league, match_date, settled, outcomes):
    """Generate a realistic 2-way prediction."""
    prob_h = random.uniform(0.35, 0.70)
    prob_a = 1.0 - prob_h

    predicted = outcomes[0] if prob_h > prob_a else outcomes[1]
    confidence = max(prob_h, prob_a)

    odds_h = round(1 / max(prob_h, 0.05) * random.uniform(0.9, 1.12), 2)
    odds_a = round(1 / max(prob_a, 0.05) * random.uniform(0.9, 1.12), 2)

    pred_odds = odds_h if predicted == outcomes[0] else odds_a
    ev = confidence * pred_odds - 1
    is_value = ev > 0.02
    kelly = max(0, ev / max(pred_odds - 1, 0.1)) * 0.25 if is_value else None

    actual = None
    is_correct = None
    if settled:
        if random.random() < 0.97:
            actual = predicted
        else:
            actual = outcomes[1] if predicted == outcomes[0] else outcomes[0]
        is_correct = actual == predicted

    return Prediction(
        sport=sport,
        match_external_id=f"{sport[:2]}_{random.randint(100000, 999999)}",
        match_date=match_date,
        home_team=home,
        away_team=away,
        league=league,
        prob_home=round(prob_h, 4),
        prob_draw=None,
        prob_away=round(prob_a, 4),
        predicted_outcome=predicted,
        confidence=round(confidence, 4),
        odds_home=odds_h,
        odds_draw=None,
        odds_away=odds_a,
        is_value_bet=is_value,
        expected_value=round(ev, 4) if is_value else None,
        kelly_stake=round(kelly * 100, 2) if kelly else None,
        actual_outcome=actual,
        is_correct=is_correct,
        model_version="v20261120_demo",
        **_ou_fields(sport, settled),
    )


async def seed():
    await init_db()

    now = datetime.now(timezone.utc)
    predictions = []

    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    def _upcoming_date(i, upcoming_start, count, today_slots=2):
        """First `today_slots` upcoming get today's date, rest are random future."""
        idx = i - upcoming_start
        if idx < today_slots:
            return today_start + timedelta(hours=random.randint(10, 22))
        return now + timedelta(days=random.randint(1, 14), hours=random.randint(8, 22))

    # --- Football: 45 settled + 15 upcoming ---
    for i in range(60):
        home, away, league = random.choice(FOOTBALL_MATCHES)
        if i < 45:
            match_date = now - timedelta(days=random.randint(1, 90), hours=random.randint(0, 12))
            settled = True
        else:
            match_date = _upcoming_date(i, 45, 15)
            settled = False
        predictions.append(_random_football_pred(home, away, league, match_date, settled))

    # --- Basketball: 40 settled + 10 upcoming ---
    for i in range(50):
        home, away, league = random.choice(BASKETBALL_MATCHES)
        if i < 40:
            match_date = now - timedelta(days=random.randint(1, 60), hours=random.randint(0, 12))
            settled = True
        else:
            match_date = _upcoming_date(i, 40, 10)
            settled = False
        predictions.append(_random_2way_pred("basketball", home, away, league, match_date, settled, TWO_WAY_OUTCOMES))

    # --- Hockey: 35 settled + 10 upcoming ---
    for i in range(45):
        home, away, league = random.choice(HOCKEY_MATCHES)
        if i < 35:
            match_date = now - timedelta(days=random.randint(1, 60), hours=random.randint(0, 12))
            settled = True
        else:
            match_date = _upcoming_date(i, 35, 10)
            settled = False
        predictions.append(_random_2way_pred("hockey", home, away, league, match_date, settled, TWO_WAY_OUTCOMES))

    # --- Tennis: 30 settled + 10 upcoming ---
    for i in range(40):
        home, away, league = random.choice(TENNIS_MATCHES)
        if i < 30:
            match_date = now - timedelta(days=random.randint(1, 45), hours=random.randint(0, 12))
            settled = True
        else:
            match_date = _upcoming_date(i, 30, 10)
            settled = False
        predictions.append(_random_2way_pred("tennis", home, away, league, match_date, settled, TENNIS_OUTCOMES))

    async with async_session() as session:
        session.add_all(predictions)
        await session.commit()

    total = len(predictions)
    settled_count = sum(1 for p in predictions if p.actual_outcome is not None)
    value_count = sum(1 for p in predictions if p.is_value_bet and p.actual_outcome is None)
    print(f"Seeded {total} predictions ({settled_count} settled, {total - settled_count} upcoming)")
    print(f"  Football:   60  |  Basketball: 50  |  Hockey: 45  |  Tennis: 40")
    print(f"  Value bets (upcoming): {value_count}")


if __name__ == "__main__":
    asyncio.run(seed())
