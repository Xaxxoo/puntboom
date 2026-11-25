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
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, mean_absolute_error
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


def _build_classifier_models(over_rate: float = 0.5) -> dict:
    """Build candidate binary classifiers for O/U prediction.

    Uses balanced class weights so models don't collapse to the base rate.
    """
    models = {
        "logistic": LogisticRegression(max_iter=1000, class_weight="balanced"),
    }
    if XGBClassifier is not None:
        # scale_pos_weight = n_negative / n_positive
        spw = (1 - over_rate) / max(over_rate, 0.01)
        models["xgboost"] = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=spw,
            verbosity=0,
        )
    if LGBMClassifier is not None:
        models["lightgbm"] = LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            objective="binary",
            is_unbalance=True,
            verbose=-1,
        )
    return models


def train_totals_classifier(
    sport: str,
    features_df: pd.DataFrame,
    labels: pd.Series,
    line: float,
) -> dict:
    """Train a binary classifier on Over/Under labels.

    Args:
        sport: Sport name (e.g. "football", "hockey").
        features_df: Feature DataFrame (same as regression).
        labels: Numeric total goals (home_score + away_score).
        line: The O/U line (e.g. 2.5 for football, 5.5 for hockey).

    Returns dict with model metadata and metrics.
    """
    model_dir = settings.model_dir / sport
    model_dir.mkdir(parents=True, exist_ok=True)

    # Convert numeric totals to binary labels: 1 = Over, 0 = Under
    y_raw = labels.dropna().astype(float)
    y = (y_raw > line).astype(int)
    X = features_df.loc[y.index]

    if len(X) < 20:
        logger.warning(f"Not enough O/U classifier data for {sport}: {len(X)} samples")
        return {"sport": sport, "error": "insufficient_data", "samples": len(X)}

    X = X.fillna(0)
    over_rate = float(y.mean())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_splits = min(5, max(2, len(X) // 20))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    candidate_models = _build_classifier_models(over_rate)
    best_name = None
    best_score = float("inf")
    best_model = None
    results = {}

    for name, model in candidate_models.items():
        try:
            scores = []
            for train_idx, val_idx in tscv.split(X_scaled):
                model.fit(X_scaled[train_idx], y.iloc[train_idx])
                proba = model.predict_proba(X_scaled[val_idx])[:, 1]
                bs = brier_score_loss(y.iloc[val_idx], proba)
                scores.append(bs)

            avg_brier = np.mean(scores)
            results[name] = {"avg_brier": avg_brier, "cv_scores": [float(s) for s in scores]}
            logger.info(f"{sport}_totals_cls/{name}: avg Brier={avg_brier:.4f}")

            if avg_brier < best_score:
                best_score = avg_brier
                best_name = name
                best_model = model
        except Exception:
            logger.exception(f"Failed to train {sport}_totals_cls/{name}")

    if best_model is None:
        return {"sport": sport, "error": "all_models_failed"}

    # Collect out-of-fold predictions from the best model for threshold tuning
    oof_proba = np.full(len(X_scaled), np.nan)
    for train_idx, val_idx in tscv.split(X_scaled):
        best_model.fit(X_scaled[train_idx], y.iloc[train_idx])
        oof_proba[val_idx] = best_model.predict_proba(X_scaled[val_idx])[:, 1]
    oof_mask = ~np.isnan(oof_proba)

    # Find the decision threshold where the predicted Over rate matches
    # the actual Over rate.  This prevents the model from collapsing to
    # "always predict Over" (which pure accuracy-maximisation tends to do
    # when probabilities are compressed near the base rate).
    actual_over_rate = float(y.values[oof_mask].mean())
    best_threshold = float(np.percentile(oof_proba[oof_mask], (1 - actual_over_rate) * 100))
    best_threshold = round(best_threshold, 3)

    # Measure accuracy at this rate-matched threshold
    oof_preds = (oof_proba[oof_mask] >= best_threshold).astype(int)
    threshold_acc = accuracy_score(y.values[oof_mask], oof_preds)
    pred_rate = float(oof_preds.mean())
    logger.info(
        f"{sport}_totals_cls: threshold={best_threshold:.3f} "
        f"(OOF accuracy={threshold_acc:.4f}, pred_over={pred_rate:.3f}, "
        f"actual_over={actual_over_rate:.3f})"
    )

    # Retrain best on all data
    best_model.fit(X_scaled, y)

    # Skip probability calibration for the O/U classifier.
    # With balanced class weights the raw model spreads probabilities
    # across a useful range (e.g. 0.35–0.83).  Calibration (isotonic or
    # sigmoid) compresses them back toward the base rate (~0.55),
    # destroying the model's ability to discriminate.  Instead we rely on
    # the optimised decision threshold found above.
    final_model = best_model

    # Save artifacts (include threshold in meta)
    meta = {
        "feature_names": list(features_df.columns),
        "line": line,
        "threshold": best_threshold,
    }
    version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    joblib.dump(final_model, model_dir / f"totals_cls_model_{version}.joblib")
    joblib.dump(scaler, model_dir / f"totals_cls_scaler_{version}.joblib")
    joblib.dump(meta, model_dir / f"totals_cls_meta_{version}.joblib")

    # Also save as "latest"
    joblib.dump(final_model, model_dir / "totals_cls_model_latest.joblib")
    joblib.dump(scaler, model_dir / "totals_cls_scaler_latest.joblib")
    joblib.dump(meta, model_dir / "totals_cls_meta_latest.joblib")

    # Final evaluation using optimal threshold
    final_proba = final_model.predict_proba(X_scaled)[:, 1]
    final_preds = (final_proba >= best_threshold).astype(int)
    final_brier = brier_score_loss(y, final_proba)
    final_acc = accuracy_score(y, final_preds)
    pred_over_rate = float(final_preds.mean())

    return {
        "sport": sport,
        "model_type": "totals_classifier",
        "best_model": best_name,
        "version": version,
        "samples": len(X),
        "line": line,
        "metrics": {
            "brier_score": float(final_brier),
            "accuracy": float(final_acc),
            "over_rate": over_rate,
            "pred_over_rate": pred_over_rate,
            "threshold": best_threshold,
        },
        "cv_results": {k: {"avg_brier": v["avg_brier"]} for k, v in results.items()},
    }
