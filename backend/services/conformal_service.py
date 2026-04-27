import logging
import uuid
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from mapie.classification import MapieClassifier
from mapie.metrics import classification_coverage_score
from sklearn.model_selection import train_test_split
from sqlalchemy.orm import Session

from sklearn.model_selection import train_test_split

from backend.config import settings
from backend.models.model_version import ModelVersion, ModelType
from backend.models.prediction import Prediction
from backend.ml.feature_engineering import get_feature_columns
from backend.services.cmapss_loader import load_train, load_test, simulate_delayed_labels
from backend.services.training_service import (
    _load_model,
    _split_features_labels,
    train_continual,
    train_baseline,
)

logger = logging.getLogger(__name__)

# Conformal prediction target coverage
# 0.90 = "90% of true labels fall within the prediction set"
TARGET_COVERAGE = 0.90


def _get_conformal_path(model_id: str) -> Path:
    return Path(settings.MODEL_STORE_PATH) / f"{model_id}_conformal.joblib"


def build_conformal_model(
    db: Session,
    model_version: ModelVersion,
    subset: str = "FD001",
    delay_cycles: int = 30,
    failure_threshold: int = 30,
    ) -> ModelVersion:
    logger.info(f"[Conformal] Building conformal wrapper for model {model_version.id}")

    base_model = _load_model(model_version.artifact_path)
    train_features = list(base_model.feature_names_in_)

    # Use FULL training data for calibration — ensures both classes present
    # Split 80/20 — calibration set gets 20% with stratification
    df = load_train(subset, failure_threshold=failure_threshold)
    X_all, y_all = _split_features_labels(df)
    X_all = X_all[[c for c in train_features if c in X_all.columns]]

    _, X_cal, _, y_cal = train_test_split(
        X_all, y_all,
        test_size=0.2,
        random_state=42,
        stratify=y_all,
    )

    class_dist = pd.Series(y_cal).value_counts().to_dict()
    logger.info(f"[Conformal] Calibration: {len(X_cal)} samples | classes: {class_dist}")

    mapie = MapieClassifier(
        estimator=base_model,
        cv="prefit",
        method="score",
    )
    mapie.fit(X_cal, y_cal)

    conformal_path = _get_conformal_path(model_version.id)
    joblib.dump(mapie, conformal_path)

    # Evaluate coverage on calibration set
    y_pred, y_pred_sets = mapie.predict(X_cal, alpha=1 - TARGET_COVERAGE)
    pred_sets = y_pred_sets[:, :, 0]

    logger.info(f"[Conformal] Sample pred_sets:\n{pred_sets[:5]}")

    coverage = float(classification_coverage_score(y_cal, pred_sets))
    avg_width = float(pred_sets.sum(axis=1).mean())

    logger.info(f"[Conformal] Coverage: {coverage:.4f} | Width: {avg_width:.4f}")

    model_version.coverage = round(coverage, 4)
    model_version.avg_interval_width = round(avg_width, 4)
    db.commit()
    db.refresh(model_version)
    return model_version


def run_predictions(
    db: Session,
    model_version: ModelVersion,
    subset: str = "FD001",
    delay_cycles: int = 30,
    failure_threshold: int = 30,
    alpha: float = 1 - TARGET_COVERAGE,
) -> list[Prediction]:
    """
    Run conformal predictions on the test set.
    Stores predictions with uncertainty intervals in the DB.

    Each prediction has:
    - predicted_label: point prediction (0 or 1)
    - lower_bound: whether class 0 is in prediction set
    - upper_bound: whether class 1 is in prediction set
    - uncertainty_score: width of prediction set (0=certain, 1=uncertain, 2=both)
    """
    logger.info(
        f"[Conformal] Running predictions for model {model_version.id}"
    )

    conformal_path = _get_conformal_path(model_version.id)
    if not conformal_path.exists():
        raise FileNotFoundError(
            f"Conformal model not found: {conformal_path}. "
            "Run build_conformal_model first."
        )

    mapie = joblib.load(conformal_path)

    # Load test data
    test_df, true_rul = load_test(subset)

    # Get last cycle per engine (one prediction per engine)
    last_cycles = test_df.groupby("engine_id").last().reset_index()
    feature_cols = get_feature_columns(last_cycles)

    # Add true binary label from RUL
    last_cycles["true_label"] = (true_rul <= failure_threshold).astype(int)

    train_features = list(mapie.estimator.feature_names_in_)
    X_test = last_cycles[[c for c in train_features if c in last_cycles.columns]]
    y_true = last_cycles["true_label"].values

    # Predict with conformal intervals
    y_pred, y_pred_sets = mapie.predict(X_test, alpha=alpha)
    # y_pred_sets shape: (n_samples, n_classes, n_alphas)
    pred_sets = y_pred_sets[:, :, 0]  # shape: (n_samples, 2)

    predictions = []
    for i, row in enumerate(last_cycles.itertuples()):
        lower = float(pred_sets[i, 0])   # class 0 in set?
        upper = float(pred_sets[i, 1])   # class 1 in set?
        uncertainty = float(pred_sets[i, 0]) + float(pred_sets[i, 1])

        pred = Prediction(
            id=str(uuid.uuid4()),
            model_version_id=model_version.id,
            engine_id=int(row.engine_id),
            cycle=int(row.cycle),
            predicted_label=int(y_pred[i]),
            lower_bound=lower,
            upper_bound=upper,
            true_label=int(y_true[i]),
            created_at=datetime.utcnow(),
        )
        predictions.append(pred)

    db.bulk_save_objects(predictions)
    db.commit()

    logger.info(
        f"[Conformal] Saved {len(predictions)} predictions for model {model_version.id}"
    )
    return predictions


def get_uncertainty_summary(
    db: Session,
    model_version_id: str,
) -> dict:
    predictions = (
        db.query(Prediction)
        .filter_by(model_version_id=model_version_id)
        .all()
    )

    if not predictions:
        return {"error": "No predictions found for this model version."}

    total = len(predictions)

    # lower_bound = 1.0 means class 0 is in prediction set
    # upper_bound = 1.0 means class 1 is in prediction set
    # Both = 1.0 means uncertain (model can't decide)
    uncertain = sum(
        1 for p in predictions
        if p.lower_bound == 1.0 and p.upper_bound == 1.0
    )
    certain_positive = sum(
        1 for p in predictions
        if p.upper_bound == 1.0 and p.lower_bound == 0.0
    )
    certain_negative = sum(
        1 for p in predictions
        if p.lower_bound == 1.0 and p.upper_bound == 0.0
    )

    return {
        "model_version_id": model_version_id,
        "total_predictions": total,
        "certain_positive": certain_positive,
        "certain_negative": certain_negative,
        "uncertain": uncertain,
        "uncertainty_rate": round(uncertain / total, 4) if total > 0 else 0.0,
    }