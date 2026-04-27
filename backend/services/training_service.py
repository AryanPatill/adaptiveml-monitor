import logging
import uuid
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sqlalchemy.orm import Session

from backend.config import settings
from backend.models.model_version import ModelVersion, ModelType
from backend.models.experiment import Experiment, ExperimentStatus
from backend.ml.feature_engineering import get_feature_columns
from backend.ml.labeling_functions import (
    apply_labeling_functions,
    generate_pseudo_labels,
    ABSTAIN,
)
from backend.services.cmapss_loader import load_train, simulate_delayed_labels

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# XGBoost hyperparameters
# Tuned for CMAPSS class imbalance (~15% positive rate)
# ─────────────────────────────────────────────
XGBOOST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 5,   # handles class imbalance
    "use_label_encoder": False,
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": -1,
}


def _get_model_path(model_id: str) -> Path:
    path = Path(settings.MODEL_STORE_PATH) / f"{model_id}.joblib"
    return path


def _save_model(model, model_id: str) -> str:
    path = _get_model_path(model_id)
    joblib.dump(model, path)
    logger.info(f"Model saved: {path}")
    return str(path)


def _load_model(artifact_path: str):
    path = Path(artifact_path)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    return joblib.load(path)


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    return {
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    }


def _split_features_labels(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    feature_cols = get_feature_columns(df)
    X = df[feature_cols]
    y = df["label"].values
    return X, y


# ─────────────────────────────────────────────
# Baseline Training
# Train ONLY on confirmed labels (no pseudo-labels)
# This is the "before" model in our comparison
# ─────────────────────────────────────────────
def train_baseline(
    db: Session,
    experiment_id: str,
    subset: str = "FD001",
    delay_cycles: int = 30,
    failure_threshold: int = 30,
) -> ModelVersion:
    logger.info(f"[Baseline] Starting training for experiment {experiment_id}")

    # Load and split data
    df = load_train(subset, failure_threshold=failure_threshold)
    confirmed, _ = simulate_delayed_labels(df, delay_cycles=delay_cycles)

    X, y = _split_features_labels(confirmed)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(
        f"[Baseline] Training on {len(X_train)} confirmed samples | "
        f"Validation: {len(X_val)} samples"
    )

    # Train
    model = XGBClassifier(**XGBOOST_PARAMS)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_val)
    metrics = _compute_metrics(y_val, y_pred)
    logger.info(f"[Baseline] Metrics: {metrics}")

    # Persist
    model_id = str(uuid.uuid4())
    artifact_path = _save_model(model, model_id)

    model_version = ModelVersion(
        id=model_id,
        experiment_id=experiment_id,
        model_type=ModelType.baseline,
        artifact_path=artifact_path,
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1_score=metrics["f1_score"],
        coverage=None,
        avg_interval_width=None,
        created_at=datetime.utcnow(),
    )
    db.add(model_version)
    db.commit()
    db.refresh(model_version)

    logger.info(f"[Baseline] ModelVersion saved: {model_id}")
    return model_version


# ─────────────────────────────────────────────
# Continual Training
# Train on confirmed labels + pseudo-labeled unconfirmed samples
# This is the "after" model in our comparison
# ─────────────────────────────────────────────
def train_continual(
    db: Session,
    experiment_id: str,
    subset: str = "FD001",
    delay_cycles: int = 30,
    failure_threshold: int = 30,
) -> ModelVersion:
    logger.info(f"[Continual] Starting training for experiment {experiment_id}")

    # Load and split data
    df = load_train(subset, failure_threshold=failure_threshold)
    confirmed, unconfirmed = simulate_delayed_labels(df, delay_cycles=delay_cycles)

    # Apply Snorkel LFs to unconfirmed samples
    label_matrix, _ = apply_labeling_functions(unconfirmed)
    pseudo_df = generate_pseudo_labels(unconfirmed, label_matrix)

    # Keep only confidently pseudo-labeled samples (drop ABSTAIN)
    pseudo_df = pseudo_df[pseudo_df["pseudo_label"] != ABSTAIN].copy()
    pseudo_df["label"] = pseudo_df["pseudo_label"]

    logger.info(
        f"[Continual] Confirmed: {len(confirmed)} | "
        f"Pseudo-labeled: {len(pseudo_df)} | "
        f"Total: {len(confirmed) + len(pseudo_df)}"
    )

    # Combine confirmed + pseudo-labeled
    combined = pd.concat(
        [confirmed, pseudo_df],
        ignore_index=True,
    )

    X, y = _split_features_labels(combined)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(
        f"[Continual] Training on {len(X_train)} samples | "
        f"Validation: {len(X_val)} samples"
    )

    # Train
    model = XGBClassifier(**XGBOOST_PARAMS)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_val)
    metrics = _compute_metrics(y_val, y_pred)
    logger.info(f"[Continual] Metrics: {metrics}")

    # Persist
    model_id = str(uuid.uuid4())
    artifact_path = _save_model(model, model_id)

    model_version = ModelVersion(
        id=model_id,
        experiment_id=experiment_id,
        model_type=ModelType.continual,
        artifact_path=artifact_path,
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1_score=metrics["f1_score"],
        coverage=None,
        avg_interval_width=None,
        created_at=datetime.utcnow(),
    )
    db.add(model_version)
    db.commit()
    db.refresh(model_version)

    logger.info(f"[Continual] ModelVersion saved: {model_id}")
    return model_version


# ─────────────────────────────────────────────
# Incremental Retrain
# Called after human labels approved from active learning queue
# Retrains continual model with newly confirmed human labels added
# ─────────────────────────────────────────────
def retrain_with_human_labels(
    db: Session,
    experiment_id: str,
    human_labeled_df: pd.DataFrame,
    subset: str = "FD001",
    delay_cycles: int = 30,
    failure_threshold: int = 30,
) -> ModelVersion:
    logger.info(f"[Retrain] Starting incremental retrain for experiment {experiment_id}")

    # Load base confirmed data
    df = load_train(subset, failure_threshold=failure_threshold)
    confirmed, unconfirmed = simulate_delayed_labels(df, delay_cycles=delay_cycles)

    # Apply pseudo labels to remaining unconfirmed
    label_matrix, _ = apply_labeling_functions(unconfirmed)
    pseudo_df = generate_pseudo_labels(unconfirmed, label_matrix)
    pseudo_df = pseudo_df[pseudo_df["pseudo_label"] != ABSTAIN].copy()
    pseudo_df["label"] = pseudo_df["pseudo_label"]

    # Combine all three sources:
    # confirmed (ground truth) + pseudo (weak) + human-reviewed (strong)
    combined = pd.concat(
        [confirmed, pseudo_df, human_labeled_df],
        ignore_index=True,
    )

    logger.info(
        f"[Retrain] Sources — confirmed: {len(confirmed)} | "
        f"pseudo: {len(pseudo_df)} | "
        f"human: {len(human_labeled_df)} | "
        f"total: {len(combined)}"
    )

    X, y = _split_features_labels(combined)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(**XGBOOST_PARAMS)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    metrics = _compute_metrics(y_val, y_pred)
    logger.info(f"[Retrain] Metrics: {metrics}")

    model_id = str(uuid.uuid4())
    artifact_path = _save_model(model, model_id)

    model_version = ModelVersion(
        id=model_id,
        experiment_id=experiment_id,
        model_type=ModelType.continual,
        artifact_path=artifact_path,
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1_score=metrics["f1_score"],
        coverage=None,
        avg_interval_width=None,
        created_at=datetime.utcnow(),
    )
    db.add(model_version)
    db.commit()
    db.refresh(model_version)

    logger.info(f"[Retrain] ModelVersion saved: {model_id}")
    return model_version


def get_model_comparison(
    db: Session,
    experiment_id: str,
) -> dict:
    """
    Fetch latest baseline and continual model versions for an experiment
    and compute improvement deltas.
    """
    baseline = (
        db.query(ModelVersion)
        .filter_by(experiment_id=experiment_id, model_type=ModelType.baseline)
        .order_by(ModelVersion.created_at.desc())
        .first()
    )

    continual = (
        db.query(ModelVersion)
        .filter_by(experiment_id=experiment_id, model_type=ModelType.continual)
        .order_by(ModelVersion.created_at.desc())
        .first()
    )

    improvement_f1 = None
    if baseline and continual and baseline.f1_score is not None and continual.f1_score is not None:
        improvement_f1 = round(continual.f1_score - baseline.f1_score, 4)

    improvement_recall = None
    if baseline and continual and baseline.recall is not None and continual.recall is not None:
        improvement_recall = round(continual.recall - baseline.recall, 4)

    return {
        "baseline": baseline,
        "continual": continual,
        "improvement_f1": improvement_f1,
        "improvement_recall": improvement_recall,
    }