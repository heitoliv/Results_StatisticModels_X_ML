from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WindowedSplits:
    X_train: np.ndarray
    y_train: np.ndarray
    X_validation: np.ndarray
    y_validation: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    train_timestamps: np.ndarray
    validation_timestamps: np.ndarray
    test_timestamps: np.ndarray


def create_windows_by_target_split(
    scaled_frames: dict,
    features: Tuple[str, ...],
    target: str,
    datetime_column: str,
    lookback: int,
    horizon: int,
) -> WindowedSplits:
    """
    Cria janelas sobre a série cronológica concatenada e atribui cada amostra
    ao split conforme o timestamp do alvo. Assim, o primeiro ponto da
    validation/test pode usar histórico observado do split anterior, sem usar futuro.
    """
    full = pd.concat(
        [scaled_frames["train"], scaled_frames["validation"], scaled_frames["test"]],
        axis=0,
        ignore_index=True,
    )
    values = full.loc[:, list(features)].to_numpy(dtype=np.float32)
    target_values = full[target].to_numpy(dtype=np.float32)
    timestamps = full[datetime_column].astype(str).to_numpy()

    train_end = len(scaled_frames["train"])
    validation_end = train_end + len(scaled_frames["validation"])

    buckets = {
        "train": ([], [], []),
        "validation": ([], [], []),
        "test": ([], [], []),
    }

    for target_idx in range(lookback, len(full) - horizon + 1):
        target_end_idx = target_idx + horizon - 1
        X = values[target_idx - lookback:target_idx]
        y = target_values[target_idx:target_idx + horizon]
        y = y[0] if horizon == 1 else y

        if target_end_idx < train_end:
            split = "train"
        elif target_end_idx < validation_end:
            split = "validation"
        else:
            split = "test"

        buckets[split][0].append(X)
        buckets[split][1].append(y)
        buckets[split][2].append(timestamps[target_end_idx])

    def finalize(bucket):
        Xs, ys, ts = bucket
        if not Xs:
            raise ValueError("Uma das partições ficou sem janelas.")
        return (
            np.asarray(Xs, dtype=np.float32),
            np.asarray(ys, dtype=np.float32),
            np.asarray(ts),
        )

    train = finalize(buckets["train"])
    validation = finalize(buckets["validation"])
    test = finalize(buckets["test"])

    return WindowedSplits(
        X_train=train[0], y_train=train[1], train_timestamps=train[2],
        X_validation=validation[0], y_validation=validation[1], validation_timestamps=validation[2],
        X_test=test[0], y_test=test[1], test_timestamps=test[2],
    )


def validate_windows(windows: WindowedSplits) -> None:
    for name, X, y in (
        ("TRAIN", windows.X_train, windows.y_train),
        ("VALIDATION", windows.X_validation, windows.y_validation),
        ("TEST", windows.X_test, windows.y_test),
    ):
        if len(X) != len(y):
            raise ValueError(f"{name}: número de amostras de X e y diferente.")
        if X.ndim != 3:
            raise ValueError(f"{name}: X deve ser 3D.")
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError(f"{name}: há NaN/Inf.")


def save_windows(windows: WindowedSplits, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "X_train", "y_train", "X_validation", "y_validation", "X_test", "y_test",
        "train_timestamps", "validation_timestamps", "test_timestamps",
    ):
        np.save(output_dir / f"{name}.npy", getattr(windows, name))
