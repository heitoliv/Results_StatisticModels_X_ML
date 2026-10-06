from __future__ import annotations

from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


SCALER_NAMES = ("minmax", "standard")


def create_scaler(name: str, minmax_range: Tuple[float, float] = (0.0, 1.0)):
    if name == "minmax":
        return MinMaxScaler(feature_range=minmax_range)
    if name == "standard":
        return StandardScaler()
    raise ValueError(f"Scaler desconhecido: {name}")


def fit_transform_splits(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    features: Tuple[str, ...],
    scaler_name: str,
    minmax_range: Tuple[float, float] = (0.0, 1.0),
):
    scaler = create_scaler(scaler_name, minmax_range)

    # FIT exclusivamente no TRAIN.
    feature_columns = list(features)
    scaler.fit(train.loc[:, feature_columns])

    transformed = {}
    for split_name, frame in (
        ("train", train),
        ("validation", validation),
        ("test", test),
    ):
        result = frame.copy()
        result.loc[:, feature_columns] = scaler.transform(frame.loc[:, feature_columns])
        transformed[split_name] = result

    return transformed, scaler


def inverse_transform_target(
    values: np.ndarray,
    scaler,
    features: Tuple[str, ...],
    target: str,
) -> np.ndarray:
    """Inverte somente o target para 1D ou (N, horizon), independentemente do scaler."""
    values = np.asarray(values, dtype=np.float64)
    original_shape = values.shape
    flat = values.reshape(-1, 1)

    if target not in features:
        raise ValueError(f"Target '{target}' não está em features.")

    target_idx = list(features).index(target)
    dummy = np.zeros((flat.shape[0], len(features)), dtype=np.float64)
    dummy[:, target_idx] = flat[:, 0]

    restored = scaler.inverse_transform(dummy)[:, target_idx]
    return restored.reshape(original_shape)


def save_processed_splits(
    transformed: dict,
    scaler,
    output_dir: Path,
    scaler_name: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    transformed["train"].to_csv(output_dir / "train.csv", index=False)
    transformed["validation"].to_csv(output_dir / "validation.csv", index=False)
    transformed["test"].to_csv(output_dir / "test.csv", index=False)
    joblib.dump(scaler, output_dir / "scaler.pkl")
