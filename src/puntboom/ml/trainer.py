"""Model training with TimeSeriesSplit CV, calibration, and persistence."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:
    XGBClassifier = None
    XGBRegressor = None

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except Exception:
    LGBMClassifier = None
    LGBMRegressor = None

from puntboom.config import settings
from puntboom.ml.evaluation import evaluate_model

logger = logging.getLogger(__name__)


SPORT_CONFIGS = {
    "football": {"n_classes": 3, "labels": ["H", "D", "A"]},
    "basketball": {"n_classes": 2, "labels": ["H", "A"]},
    "hockey": {"n_classes": 2, "labels": ["H", "A"]},
    "tennis": {"n_classes": 2, "labels": ["1", "2"]},
}


def _build_models(n_classes: int) -> dict:
    """Build candidate models for a sport."""
    models = {
        "logistic": LogisticRegression(max_iter=1000),
    }
    if XGBClassifier is not None:
        models["xgboost"] = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            objective="multi:softprob" if n_classes > 2 else "binary:logistic",
            eval_metric="mlogloss" if n_classes > 2 else "logloss",
            verbosity=0,
        )
    if LGBMClassifier is not None:
        models["lightgbm"] = LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            objective="multiclass" if n_classes > 2 else "binary",
            verbose=-1,
        )
    return models


def train_sport_model(
    sport: str,
    features_df: pd.DataFrame,
    labels: pd.Series,
) -> dict:
    """Train models for a sport, select best, calibrate, and save.

    Returns dict with model metadata and metrics.
    """
    config = SPORT_CONFIGS[sport]
    model_dir = settings.model_dir / sport
    model_dir.mkdir(parents=True, exist_ok=True)

    # Encode labels
    label_map = {label: i for i, label in enumerate(config["labels"])}
    y = labels.map(label_map).dropna().astype(int)
    X = features_df.loc[y.index]

    if len(X) < 20:
        logger.warning(f"Not enough data for {sport}: {len(X)} samples")
        return {"sport": sport, "error": "insufficient_data", "samples": len(X)}

    # Fill NaN features with 0
    X = X.fillna(0)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Time series cross-validation
    n_splits = min(5, max(2, len(X) // 20))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    candidate_models = _build_models(config["n_classes"])
    best_name = None
    best_score = float("inf")
    best_model = None
    results = {}

    for name, model in candidate_models.items():
        try:
            scores = []
            for train_idx, val_idx in tscv.split(X_scaled):
                model.fit(X_scaled[train_idx], y.iloc[train_idx])
                probs = model.predict_proba(X_scaled[val_idx])
                metrics = evaluate_model(
                    y.iloc[val_idx].values, probs, config["labels"]
                )
                scores.append(metrics["brier_score"])

            avg_brier = np.mean(scores)
            results[name] = {"avg_brier": avg_brier, "cv_scores": scores}
            logger.info(f"{sport}/{name}: avg Brier={avg_brier:.4f}")

            if avg_brier < best_score:
                best_score = avg_brier
                best_name = name
                best_model = model
        except Exception:
            logger.exception(f"Failed to train {sport}/{name}")

    if best_model is None:
        return {"sport": sport, "error": "all_models_failed"}

    # Retrain best model on all data
    best_model.fit(X_scaled, y)

    # Calibrate probabilities
    calibrated = CalibratedClassifierCV(best_model, cv=3, method="isotonic")
    try:
        calibrated.fit(X_scaled, y)
        final_model = calibrated
    except Exception:
        logger.warning(f"Calibration failed for {sport}, using uncalibrated model")
        final_model = best_model

    # Save artifacts
    version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    joblib.dump(final_model, model_dir / f"model_{version}.joblib")
    joblib.dump(scaler, model_dir / f"scaler_{version}.joblib")
    joblib.dump(
        {"labels": config["labels"], "feature_names": list(features_df.columns)},
        model_dir / f"meta_{version}.joblib",
    )

    # Also save as "latest"
    joblib.dump(final_model, model_dir / "model_latest.joblib")
    joblib.dump(scaler, model_dir / "scaler_latest.joblib")
    joblib.dump(
        {"labels": config["labels"], "feature_names": list(features_df.columns)},
        model_dir / "meta_latest.joblib",
    )

    # Final evaluation on full dataset
    final_probs = final_model.predict_proba(X_scaled)
    final_metrics = evaluate_model(y.values, final_probs, config["labels"])

    return {
        "sport": sport,
        "best_model": best_name,
        "version": version,
        "samples": len(X),
        "metrics": final_metrics,
        "cv_results": {k: {"avg_brier": v["avg_brier"]} for k, v in results.items()},
    }


def _build_regression_models() -> dict:
    """Build candidate regression models for totals prediction."""
    models = {
        "ridge": Ridge(alpha=1.0),
    }
    if XGBRegressor is not None:
        models["xgboost"] = XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            objective="reg:squarederror",
            verbosity=0,
        )
    if LGBMRegressor is not None:
        models["lightgbm"] = LGBMRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            objective="regression",
            verbose=-1,
        )
    return models


def train_totals_model(
    sport: str,
    features_df: pd.DataFrame,
    labels: pd.Series,
) -> dict:
    """Train a regression model to predict total goals/points.

    Labels should be numeric (e.g. home_score + away_score).
    Returns dict with model metadata and metrics.
    """
    model_dir = settings.model_dir / sport
    model_dir.mkdir(parents=True, exist_ok=True)

    y = labels.dropna().astype(float)
    X = features_df.loc[y.index]

    if len(X) < 20:
        logger.warning(f"Not enough totals data for {sport}: {len(X)} samples")
        return {"sport": sport, "error": "insufficient_data", "samples": len(X)}

    X = X.fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_splits = min(5, max(2, len(X) // 20))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    candidate_models = _build_regression_models()
    best_name = None
    best_mae = float("inf")
    best_model = None
    results = {}

    for name, model in candidate_models.items():
        try:
            scores = []
            for train_idx, val_idx in tscv.split(X_scaled):
                model.fit(X_scaled[train_idx], y.iloc[train_idx])
                preds = model.predict(X_scaled[val_idx])
                mae = mean_absolute_error(y.iloc[val_idx], preds)
                scores.append(mae)

            avg_mae = np.mean(scores)
            results[name] = {"avg_mae": avg_mae, "cv_scores": [float(s) for s in scores]}
            logger.info(f"{sport}_totals/{name}: avg MAE={avg_mae:.4f}")

            if avg_mae < best_mae:
                best_mae = avg_mae
                best_name = name
                best_model = model
        except Exception:
            logger.exception(f"Failed to train {sport}_totals/{name}")

    if best_model is None:
        return {"sport": sport, "error": "all_models_failed"}

    # Retrain best on all data
    best_model.fit(X_scaled, y)

    # Save artifacts
    version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    joblib.dump(best_model, model_dir / f"totals_model_{version}.joblib")
    joblib.dump(scaler, model_dir / f"totals_scaler_{version}.joblib")
    joblib.dump(
        {"feature_names": list(features_df.columns)},
        model_dir / f"totals_meta_{version}.joblib",
    )

    # Also save as "latest"
    joblib.dump(best_model, model_dir / "totals_model_latest.joblib")
    joblib.dump(scaler, model_dir / "totals_scaler_latest.joblib")
    joblib.dump(
        {"feature_names": list(features_df.columns)},
        model_dir / "totals_meta_latest.joblib",
    )

    # Final evaluation
    final_preds = best_model.predict(X_scaled)
    final_mae = mean_absolute_error(y, final_preds)
    avg_total = float(y.mean())
    std_total = float(y.std())

    return {
        "sport": sport,
        "model_type": "totals",
        "best_model": best_name,
        "version": version,
        "samples": len(X),
        "metrics": {
            "mae": float(final_mae),
            "avg_total_goals": avg_total,
            "std_total_goals": std_total,
        },
        "cv_results": {k: {"avg_mae": v["avg_mae"]} for k, v in results.items()},
    }
