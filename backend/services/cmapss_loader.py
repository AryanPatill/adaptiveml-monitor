import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

from backend.config import settings
from backend.ml.feature_engineering import (
    CMAPSS_COLUMNS,
    build_features,
    build_test_features,
    get_feature_columns,
)

logger = logging.getLogger(__name__)

VALID_SUBSETS = ["FD001", "FD002", "FD003", "FD004"]


def _get_data_path() -> Path:
    return Path(settings.CMAPSS_DATA_PATH)


def _load_raw(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"CMAPSS file not found: {file_path}")
    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=CMAPSS_COLUMNS,
        engine="python",
    )
    return df


def load_train(subset: str = "FD001", failure_threshold: int = 30) -> pd.DataFrame:
    if subset not in VALID_SUBSETS:
        raise ValueError(f"Invalid subset '{subset}'. Must be one of {VALID_SUBSETS}")
    path = _get_data_path() / f"train_{subset}.txt"
    logger.info(f"Loading CMAPSS train data: {path}")
    df = _load_raw(path)
    df = build_features(df, failure_threshold=failure_threshold)
    logger.info(
        f"Loaded train_{subset}: {len(df)} rows | "
        f"Failure rate: {df['label'].mean():.2%}"
    )
    return df


def load_test(subset: str = "FD001") -> tuple[pd.DataFrame, np.ndarray]:
    if subset not in VALID_SUBSETS:
        raise ValueError(f"Invalid subset '{subset}'. Must be one of {VALID_SUBSETS}")
    test_path = _get_data_path() / f"test_{subset}.txt"
    rul_path = _get_data_path() / f"RUL_{subset}.txt"
    logger.info(f"Loading CMAPSS test data: {test_path}")
    df = _load_raw(test_path)
    df = build_test_features(df)
    true_rul = pd.read_csv(rul_path, header=None, names=["RUL"])["RUL"].values
    logger.info(f"Loaded test_{subset}: {len(df)} rows")
    return df, true_rul


def get_dataset_summary(subset: str = "FD001") -> dict:
    if subset not in VALID_SUBSETS:
        raise ValueError(f"Invalid subset '{subset}'. Must be one of {VALID_SUBSETS}")
    train_df = load_train(subset)
    feature_cols = get_feature_columns(train_df)
    n_engines = train_df["engine_id"].nunique()
    n_cycles_total = len(train_df)
    failure_rate = train_df["label"].mean()
    avg_life = train_df.groupby("engine_id")["cycle"].max().mean()
    sensor_stats = (
        train_df[feature_cols]
        .describe()
        .loc[["mean", "std", "min", "max"]]
        .round(4)
        .to_dict()
    )
    return {
        "subset": subset,
        "n_engines": int(n_engines),
        "n_cycles_total": int(n_cycles_total),
        "n_features": len(feature_cols),
        "failure_rate": round(float(failure_rate), 4),
        "avg_engine_life_cycles": round(float(avg_life), 2),
        "feature_columns": feature_cols,
        "sensor_stats": sensor_stats,
    }


def simulate_delayed_labels(
    df: pd.DataFrame,
    delay_cycles: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    confirmed_list = []
    unconfirmed_list = []
    for engine_id, group in df.groupby("engine_id"):
        max_cycle = group["cycle"].max()
        cutoff = max_cycle - delay_cycles
        confirmed = group[group["cycle"] <= cutoff].copy()
        unconfirmed = group[group["cycle"] > cutoff].copy()
        confirmed_list.append(confirmed)
        unconfirmed_list.append(unconfirmed)
    confirmed_df = pd.concat(confirmed_list, ignore_index=True)
    unconfirmed_df = pd.concat(unconfirmed_list, ignore_index=True)
    logger.info(
        f"Label delay simulation (delay={delay_cycles} cycles): "
        f"confirmed={len(confirmed_df)} | unconfirmed={len(unconfirmed_df)}"
    )
    return confirmed_df, unconfirmed_df