import logging
import numpy as np
import pandas as pd
from snorkel.labeling import labeling_function, PandasLFApplier, LFAnalysis

logger = logging.getLogger(__name__)

ABSTAIN = -1
NEGATIVE = 0
POSITIVE = 1


@labeling_function()
def lf_sensor2_zscore(row) -> int:
    val = row.get("sensor_2_roll_mean", None)
    if val is None or np.isnan(val):
        return ABSTAIN
    if val < 643.0:
        return POSITIVE
    if val > 643.6:
        return NEGATIVE
    return ABSTAIN


@labeling_function()
def lf_rul_window_heuristic(row) -> int:
    cycle_norm = row.get("cycle_norm", None)
    if cycle_norm is None or np.isnan(cycle_norm):
        return ABSTAIN
    if cycle_norm >= 0.85:
        return POSITIVE
    if cycle_norm <= 0.50:
        return NEGATIVE
    return ABSTAIN


@labeling_function()
def lf_sensor11_spike(row) -> int:
    val = row.get("sensor_11_roll_std", None)
    if val is None or np.isnan(val):
        return ABSTAIN
    if val > 0.15:
        return POSITIVE
    if val < 0.05:
        return NEGATIVE
    return ABSTAIN


ALL_LFS = [
    lf_sensor2_zscore,
    lf_rul_window_heuristic,
    lf_sensor11_spike,
]


def apply_labeling_functions(df: pd.DataFrame) -> tuple[np.ndarray, dict]:
    logger.info(f"Applying {len(ALL_LFS)} labeling functions to {len(df)} samples...")

    applier = PandasLFApplier(lfs=ALL_LFS)
    label_matrix = applier.apply(df)

    analysis = LFAnalysis(L=label_matrix, lfs=ALL_LFS)
    analysis_df = analysis.lf_summary()

    coverage_stats = {
        "lf_names": [lf.name for lf in ALL_LFS],
        "coverage": analysis_df["Coverage"].tolist(),
        "conflict": analysis_df["Conflicts"].tolist(),
        "overlap": analysis_df["Overlaps"].tolist(),
        "n_samples": len(df),
        "n_lfs": len(ALL_LFS),
    }

    logger.info(f"LF coverage: {coverage_stats['coverage']}")
    return label_matrix, coverage_stats


def generate_pseudo_labels(
    df: pd.DataFrame,
    label_matrix: np.ndarray,
    confidence_threshold: float = 0.60,
) -> pd.DataFrame:
    df = df.copy()
    n_samples, n_lfs = label_matrix.shape
    pseudo_labels = []
    pseudo_confidences = []

    for i in range(n_samples):
        row_votes = label_matrix[i]
        non_abstain = row_votes[row_votes != ABSTAIN]

        if len(non_abstain) == 0:
            pseudo_labels.append(ABSTAIN)
            pseudo_confidences.append(0.0)
            continue

        participation_rate = len(non_abstain) / n_lfs
        if participation_rate < confidence_threshold:
            pseudo_labels.append(ABSTAIN)
            pseudo_confidences.append(float(participation_rate))
            continue

        vote = int(np.round(non_abstain.mean()))
        confidence = float(participation_rate)
        pseudo_labels.append(vote)
        pseudo_confidences.append(confidence)

    df["pseudo_label"] = pseudo_labels
    df["pseudo_confidence"] = pseudo_confidences

    labeled = (df["pseudo_label"] != ABSTAIN).sum()
    logger.info(
        f"Pseudo-labeling complete: {labeled}/{n_samples} samples labeled "
        f"({labeled/n_samples:.1%} coverage)"
    )
    return df


def get_lf_summary(df: pd.DataFrame) -> dict:
    label_matrix, coverage_stats = apply_labeling_functions(df)
    pseudo_df = generate_pseudo_labels(df, label_matrix)

    labeled_count = (pseudo_df["pseudo_label"] != ABSTAIN).sum()
    positive_count = (pseudo_df["pseudo_label"] == POSITIVE).sum()
    negative_count = (pseudo_df["pseudo_label"] == NEGATIVE).sum()

    return {
        "coverage_stats": coverage_stats,
        "pseudo_label_summary": {
            "total_samples": len(df),
            "labeled": int(labeled_count),
            "unlabeled": int(len(df) - labeled_count),
            "positive": int(positive_count),
            "negative": int(negative_count),
            "labeling_rate": round(float(labeled_count / len(df)), 4),
        },
    }