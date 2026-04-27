import logging
import uuid
from datetime import datetime

import pandas as pd
from sqlalchemy.orm import Session

from backend.models.label_queue import LabelQueue, LabelStatus
from backend.models.prediction import Prediction
from backend.models.model_version import ModelVersion
from backend.ml.feature_engineering import get_feature_columns
from backend.services.cmapss_loader import load_train, simulate_delayed_labels
from backend.services.conformal_service import build_conformal_model, run_predictions

logger = logging.getLogger(__name__)

QUEUE_SIZE = 10


def populate_label_queue(
    db: Session,
    experiment_id: str,
    model_version: ModelVersion,
    subset: str = "FD001",
    delay_cycles: int = 30,
    failure_threshold: int = 30,
) -> list[LabelQueue]:
    """
    Select top-K most uncertain samples from conformal predictions
    and store them in the label_queue table for human review.

    Uncertain = both class 0 and class 1 in prediction set
    (lower_bound=1.0 AND upper_bound=1.0)

    Idempotent: clears existing pending queue for this experiment first.
    """
    logger.info(f"[ActiveLearning] Populating queue for experiment {experiment_id}")

    # Clear existing pending items for this experiment
    db.query(LabelQueue).filter_by(
        experiment_id=experiment_id,
        status=LabelStatus.pending,
    ).delete()
    db.commit()

    # Get uncertain predictions for this model version
    uncertain_preds = (
        db.query(Prediction)
        .filter_by(model_version_id=model_version.id)
        .filter(
            Prediction.lower_bound == 1.0,
            Prediction.upper_bound == 1.0,
        )
        .limit(QUEUE_SIZE)
        .all()
    )

    if not uncertain_preds:
        logger.warning(
            f"[ActiveLearning] No uncertain predictions found for model {model_version.id}"
        )
        return []

    # Load unconfirmed sensor data to attach feature snapshots
    df = load_train(subset, failure_threshold=failure_threshold)
    _, unconfirmed = simulate_delayed_labels(df, delay_cycles=delay_cycles)
    feature_cols = get_feature_columns(unconfirmed)

    queue_items = []
    for pred in uncertain_preds:
        # Find matching row in unconfirmed by engine_id + cycle
        match = unconfirmed[
            (unconfirmed["engine_id"] == pred.engine_id) &
            (unconfirmed["cycle"] == pred.cycle)
        ]

        if match.empty:
            # Fallback: use last cycle for this engine
            match = unconfirmed[unconfirmed["engine_id"] == pred.engine_id].tail(1)

        if match.empty:
            logger.warning(
                f"[ActiveLearning] No matching row for engine {pred.engine_id} "
                f"cycle {pred.cycle} — skipping"
            )
            continue

        features_snapshot = match.iloc[0][feature_cols].to_dict()
        # Ensure all values are JSON-serializable
        features_snapshot = {
            k: float(v) for k, v in features_snapshot.items()
        }

        item = LabelQueue(
            id=str(uuid.uuid4()),
            experiment_id=experiment_id,
            engine_id=pred.engine_id,
            cycle=pred.cycle,
            features_json=features_snapshot,
            pseudo_label=pred.predicted_label,
            uncertainty_score=float(
                pred.lower_bound + pred.upper_bound
            ),
            human_label=None,
            reviewed_at=None,
            status=LabelStatus.pending,
        )
        queue_items.append(item)

    db.bulk_save_objects(queue_items)
    db.commit()

    logger.info(
        f"[ActiveLearning] Queue populated: {len(queue_items)} items"
    )
    return queue_items


def get_pending_queue(
    db: Session,
    experiment_id: str,
) -> list[LabelQueue]:
    """Return all pending items in the label queue for an experiment."""
    return (
        db.query(LabelQueue)
        .filter_by(experiment_id=experiment_id, status=LabelStatus.pending)
        .order_by(LabelQueue.uncertainty_score.desc())
        .all()
    )


def submit_human_label(
    db: Session,
    label_id: str,
    human_label: int,
) -> LabelQueue:
    """
    Submit a binary human label for a queued sample.

    Args:
        label_id: LabelQueue item ID
        human_label: 0 (no failure) or 1 (failure)

    Returns:
        Updated LabelQueue item

    Raises:
        ValueError: if item not found or already reviewed (409)
    """
    item = db.query(LabelQueue).filter_by(id=label_id).first()

    if item is None:
        raise ValueError(f"Label queue item not found: {label_id}")

    if item.status == LabelStatus.reviewed:
        raise ValueError(
            f"Item {label_id} already reviewed at {item.reviewed_at}. "
            "Cannot re-label."
        )

    if human_label not in (0, 1):
        raise ValueError(f"Invalid human_label: {human_label}. Must be 0 or 1.")

    item.human_label = human_label
    item.reviewed_at = datetime.utcnow()
    item.status = LabelStatus.reviewed

    db.commit()
    db.refresh(item)

    logger.info(
        f"[ActiveLearning] Label submitted — item: {label_id} | "
        f"label: {human_label}"
    )
    return item


def get_reviewed_as_dataframe(
    db: Session,
    experiment_id: str,
) -> pd.DataFrame:
    """
    Fetch all reviewed label queue items and return as a
    feature DataFrame ready for retraining.
    """
    reviewed = (
        db.query(LabelQueue)
        .filter_by(experiment_id=experiment_id, status=LabelStatus.reviewed)
        .all()
    )

    if not reviewed:
        return pd.DataFrame()

    rows = []
    for item in reviewed:
        row = dict(item.features_json)
        row["label"] = item.human_label
        rows.append(row)

    df = pd.DataFrame(rows)
    logger.info(
        f"[ActiveLearning] Loaded {len(df)} reviewed samples for retraining"
    )
    return df


def get_queue_stats(
    db: Session,
    experiment_id: str,
) -> dict:
    """Summary stats for the label queue — used by the UI."""
    total = db.query(LabelQueue).filter_by(experiment_id=experiment_id).count()
    pending = db.query(LabelQueue).filter_by(
        experiment_id=experiment_id, status=LabelStatus.pending
    ).count()
    reviewed = db.query(LabelQueue).filter_by(
        experiment_id=experiment_id, status=LabelStatus.reviewed
    ).count()

    reviewed_items = (
        db.query(LabelQueue)
        .filter_by(experiment_id=experiment_id, status=LabelStatus.reviewed)
        .all()
    )

    positive_labels = sum(
        1 for i in reviewed_items if i.human_label == 1
    )
    negative_labels = sum(
        1 for i in reviewed_items if i.human_label == 0
    )

    return {
        "experiment_id": experiment_id,
        "total": total,
        "pending": pending,
        "reviewed": reviewed,
        "positive_labels": positive_labels,
        "negative_labels": negative_labels,
        "review_rate": round(reviewed / total, 4) if total > 0 else 0.0,
    }