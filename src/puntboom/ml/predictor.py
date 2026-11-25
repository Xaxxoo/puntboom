"""Model loading and prediction generation."""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import poisson

from puntboom.config import settings

logger = logging.getLogger(__name__)


class SportPredictor:
    """Load a trained model and generate predictions."""

    def __init__(self, sport: str):
        self.sport = sport
        self.model = None
        self.scaler = None
        self.meta = None
        self._loaded = False

    def load(self, version: str = "latest") -> bool:
        """Load model artifacts from disk."""
        model_dir = settings.model_dir / self.sport
        try:
            self.model = joblib.load(model_dir / f"model_{version}.joblib")
            self.scaler = joblib.load(model_dir / f"scaler_{version}.joblib")
            self.meta = joblib.load(model_dir / f"meta_{version}.joblib")
            self._loaded = True
            logger.info(f"Loaded {self.sport} model (version={version})")
            return True
        except FileNotFoundError:
            logger.warning(f"No model found for {self.sport} (version={version})")
            return False

    @property
    def labels(self) -> list[str]:
        return self.meta["labels"] if self.meta else []

    @property
    def feature_names(self) -> list[str]:
        return self.meta["feature_names"] if self.meta else []

    def predict(self, features: dict[str, float]) -> dict[str, float] | None:
        """Predict probabilities for a single match.

        Returns dict mapping outcome labels to probabilities.
        """
        if not self._loaded:
            if not self.load():
                return None

        # Build feature vector in correct order
        feature_vector = []
        for name in self.feature_names:
            feature_vector.append(features.get(name, 0.0))

        X = np.array([feature_vector])
        X_scaled = self.scaler.transform(X)
        probs = self.model.predict_proba(X_scaled)[0]

        result = {}
        for label, prob in zip(self.labels, probs):
            result[label] = float(prob)
        return result

    def predict_batch(
        self, features_list: list[dict[str, float]]
    ) -> list[dict[str, float]]:
        """Predict probabilities for multiple matches."""
        if not self._loaded:
            if not self.load():
                return []

        rows = []
        for features in features_list:
            row = [features.get(name, 0.0) for name in self.feature_names]
            rows.append(row)

        X = np.array(rows)
        X_scaled = self.scaler.transform(X)
        all_probs = self.model.predict_proba(X_scaled)

        results = []
        for probs in all_probs:
            result = {}
            for label, prob in zip(self.labels, probs):
                result[label] = float(prob)
            results.append(result)
        return results


class TotalsPredictor:
    """Load a trained regression model and predict expected total goals."""

    def __init__(self, sport: str):
        self.sport = sport
        self.model = None
        self.scaler = None
        self.meta = None
        self._loaded = False

    def load(self, version: str = "latest") -> bool:
        model_dir = settings.model_dir / self.sport
        try:
            self.model = joblib.load(model_dir / f"totals_model_{version}.joblib")
            self.scaler = joblib.load(model_dir / f"totals_scaler_{version}.joblib")
            self.meta = joblib.load(model_dir / f"totals_meta_{version}.joblib")
            self._loaded = True
            logger.info(f"Loaded {self.sport} totals model (version={version})")
            return True
        except FileNotFoundError:
            logger.warning(f"No totals model found for {self.sport} (version={version})")
            return False

    @property
    def feature_names(self) -> list[str]:
        return self.meta["feature_names"] if self.meta else []

    def predict(
        self, features: dict[str, float], line: float = 2.5
    ) -> dict[str, float] | None:
        """Predict expected total goals for a single match.

        Args:
            features: Feature dictionary for the match.
            line: The total line to evaluate O/U against (2.5 for football,
                  5.5 for hockey, etc.).

        Returns dict with 'expected_total', 'prob_over', 'prob_under'.
        Uses Poisson CDF for O/U probability estimation.
        """
        if not self._loaded:
            if not self.load():
                return None

        feature_vector = [features.get(name, 0.0) for name in self.feature_names]
        X = np.array([feature_vector])
        X_scaled = self.scaler.transform(X)
        expected = float(self.model.predict(X_scaled)[0])

        # Clamp to reasonable range
        expected = max(0.0, min(expected, 15.0))

        # Estimate O/U probability using Poisson CDF
        # Poisson is the correct model for goal-scoring: variance = mean,
        # so lower predicted totals naturally spread more probability below the line.
        prob_over = 1.0 - poisson.cdf(int(line), expected)
        prob_under = 1.0 - prob_over

        return {
            "expected_total": round(expected, 2),
            "prob_over": round(float(prob_over), 3),
            "prob_under": round(float(prob_under), 3),
        }


class TotalsClassifier:
    """Load a trained binary O/U classifier and predict over/under probabilities."""

    def __init__(self, sport: str):
        self.sport = sport
        self.model = None
        self.scaler = None
        self.meta = None
        self._loaded = False

    def load(self, version: str = "latest") -> bool:
        model_dir = settings.model_dir / self.sport
        try:
            self.model = joblib.load(model_dir / f"totals_cls_model_{version}.joblib")
            self.scaler = joblib.load(model_dir / f"totals_cls_scaler_{version}.joblib")
            self.meta = joblib.load(model_dir / f"totals_cls_meta_{version}.joblib")
            self._loaded = True
            logger.info(f"Loaded {self.sport} totals classifier (version={version})")
            return True
        except FileNotFoundError:
            logger.warning(f"No totals classifier found for {self.sport} (version={version})")
            return False

    @property
    def feature_names(self) -> list[str]:
        return self.meta["feature_names"] if self.meta else []

    @property
    def threshold(self) -> float:
        """Optimal decision threshold from training (defaults to 0.5)."""
        if self.meta and "threshold" in self.meta:
            return self.meta["threshold"]
        return 0.5

    def predict(self, features: dict[str, float]) -> dict[str, float] | None:
        """Predict O/U probabilities for a single match.

        Returns dict with 'prob_over', 'prob_under', 'prediction', 'threshold'.
        Uses the optimal threshold determined during training.
        """
        if not self._loaded:
            if not self.load():
                return None

        feature_vector = [features.get(name, 0.0) for name in self.feature_names]
        X = np.array([feature_vector])
        X_scaled = self.scaler.transform(X)
        proba = self.model.predict_proba(X_scaled)[0]

        # Classes are [0=Under, 1=Over]
        prob_over = float(proba[1])
        prob_under = float(proba[0])
        t = self.threshold
        prediction = "Over" if prob_over >= t else "Under"

        return {
            "prob_over": round(prob_over, 3),
            "prob_under": round(prob_under, 3),
            "prediction": prediction,
            "threshold": t,
        }


class EnsemblePredictor:
    """Average predictions from multiple model versions."""

    def __init__(self, sport: str, versions: list[str] | None = None):
        self.sport = sport
        self.predictors: list[SportPredictor] = []

        if versions:
            for v in versions:
                p = SportPredictor(sport)
                if p.load(v):
                    self.predictors.append(p)
        else:
            # Default: just use latest
            p = SportPredictor(sport)
            if p.load():
                self.predictors.append(p)

    def predict(self, features: dict[str, float]) -> dict[str, float] | None:
        """Ensemble prediction by averaging probabilities."""
        if not self.predictors:
            return None

        all_preds = []
        for predictor in self.predictors:
            pred = predictor.predict(features)
            if pred:
                all_preds.append(pred)

        if not all_preds:
            return None

        # Average probabilities
        labels = list(all_preds[0].keys())
        result = {}
        for label in labels:
            result[label] = sum(p[label] for p in all_preds) / len(all_preds)
        return result
