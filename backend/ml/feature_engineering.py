import pandas as pd
import numpy as np

CMAPSS_COLUMNS = [
    "engine_id", "cycle",
    "op_setting_1", "op_setting_2", "op_setting_3",
    "sensor_1", "sensor_2", "sensor_3", "sensor_4", "sensor_5",
    "sensor_6", "sensor_7", "sensor_8", "sensor_9", "sensor_10",
    "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
    "sensor_16", "sensor_17", "sensor_18", "sensor_19", "sensor_20",
    "sensor_21",
]

SENSORS_TO_DROP = [
    "sensor_1", "sensor_5", "sensor_6",
    "sensor_10", "sensor_16", "sensor_18", "sensor_19",
]

ROLLING_WINDOW = 5


def add_rul_column(df: pd.DataFrame) -> pd.DataFrame:
    max_cycles = df.groupby("engine_id")["cycle"].max().reset_index()
    max_cycles.columns = ["engine_id", "max_cycle"]
    df = df.merge(max_cycles, on="engine_id", how="left")
    df["RUL"] = df["max_cycle"] - df["cycle"]
    df.drop(columns=["max_cycle"], inplace=True)
    return df


def add_binary_label(df: pd.DataFrame, failure_threshold: int = 30) -> pd.DataFrame:
    df["label"] = (df["RUL"] <= failure_threshold).astype(int)
    return df


def drop_low_variance_sensors(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [c for c in SENSORS_TO_DROP if c in df.columns]
    return df.drop(columns=cols_to_drop)


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    sensor_cols = [
        c for c in df.columns
        if c.startswith("sensor_") and c not in SENSORS_TO_DROP
    ]
    df = df.sort_values(["engine_id", "cycle"]).copy()
    for col in sensor_cols:
        df[f"{col}_roll_mean"] = (
            df.groupby("engine_id")[col]
            .transform(lambda x: x.rolling(ROLLING_WINDOW, min_periods=1).mean())
        )
        df[f"{col}_roll_std"] = (
            df.groupby("engine_id")[col]
            .transform(lambda x: x.rolling(ROLLING_WINDOW, min_periods=1).std().fillna(0))
        )
    return df


def add_cycle_norm(df: pd.DataFrame) -> pd.DataFrame:
    max_cycles = df.groupby("engine_id")["cycle"].transform("max")
    df["cycle_norm"] = df["cycle"] / max_cycles
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {
        "engine_id", "cycle", "RUL", "label",
        "op_setting_1", "op_setting_2", "op_setting_3",
    }
    return [c for c in df.columns if c not in exclude]


def build_features(df: pd.DataFrame, failure_threshold: int = 30) -> pd.DataFrame:
    df = drop_low_variance_sensors(df)
    df = add_rolling_features(df)
    df = add_cycle_norm(df)
    df = add_rul_column(df)
    df = add_binary_label(df, failure_threshold)
    df = df.dropna()
    return df


def build_test_features(df: pd.DataFrame) -> pd.DataFrame:
    df = drop_low_variance_sensors(df)
    df = add_rolling_features(df)
    df = add_cycle_norm(df)
    df = df.dropna()
    return df