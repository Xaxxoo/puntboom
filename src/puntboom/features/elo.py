"""Elo rating system shared across sports."""

import math


class EloRatingSystem:
    """Standard Elo rating system with configurable K-factor and home advantage."""

    def __init__(
        self,
        k_factor: float = 32.0,
        home_advantage: float = 50.0,
        default_rating: float = 1500.0,
    ):
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.default_rating = default_rating

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score for player/team A."""
        return 1.0 / (1.0 + math.pow(10, (rating_b - rating_a) / 400.0))

    def expected_home_win(self, home_rating: float, away_rating: float) -> float:
        """Expected score for home team (with home advantage)."""
        return self.expected_score(
            home_rating + self.home_advantage, away_rating
        )

    def update_ratings(
        self,
        home_rating: float,
        away_rating: float,
        home_actual: float,
    ) -> tuple[float, float]:
        """Update ratings after a match.

        Args:
            home_rating: Home team's current rating.
            away_rating: Away team's current rating.
            home_actual: Actual score for home team (1.0=win, 0.5=draw, 0.0=loss).

        Returns:
            Tuple of (new_home_rating, new_away_rating).
        """
        expected_home = self.expected_home_win(home_rating, away_rating)
        expected_away = 1.0 - expected_home

        new_home = home_rating + self.k_factor * (home_actual - expected_home)
        new_away = away_rating + self.k_factor * ((1.0 - home_actual) - expected_away)

        return new_home, new_away

    def win_probability(
        self, home_rating: float, away_rating: float
    ) -> dict[str, float]:
        """Get win probabilities. Returns dict with 'home', 'draw', 'away' keys."""
        exp_home = self.expected_home_win(home_rating, away_rating)
        # Simple draw probability model: higher when teams are closer
        rating_diff = abs(home_rating - away_rating)
        draw_factor = max(0.0, 0.28 - rating_diff / 2000.0)
        p_home = exp_home * (1.0 - draw_factor)
        p_away = (1.0 - exp_home) * (1.0 - draw_factor)
        p_draw = 1.0 - p_home - p_away
        return {"home": p_home, "draw": max(0.0, p_draw), "away": p_away}

    def win_probability_2way(
        self, home_rating: float, away_rating: float
    ) -> dict[str, float]:
        """Get 2-way win probabilities (no draw). For basketball, hockey, tennis."""
        exp_home = self.expected_home_win(home_rating, away_rating)
        return {"home": exp_home, "away": 1.0 - exp_home}
