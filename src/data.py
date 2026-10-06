from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd


@dataclass(frozen=True)
class SplitFrames:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def load_dataset(path: Path, datetime_column: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {path}")

    df = pd.read_csv(path)
    if datetime_column not in df.columns:
        raise ValueError(f"Coluna temporal '{datetime_column}' não encontrada em {path}")

    df[datetime_column] = pd.to_datetime(df[datetime_column], errors="raise")
    df = df.sort_values(datetime_column).drop_duplicates(subset=[datetime_column]).reset_index(drop=True)
    return df


def validate_columns(df: pd.DataFrame, columns: Tuple[str, ...]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes: {missing}")


def split_chronologically(
    df: pd.DataFrame,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> SplitFrames:
    total = train_ratio + validation_ratio + test_ratio
    if abs(total - 1.0) > 1e-12:
        raise ValueError(f"As proporções devem somar 1.0. Obtido: {total}")

    n = len(df)
    train_end = int(n * train_ratio)
    validation_end = train_end + int(n * validation_ratio)

    train = df.iloc[:train_end].copy()
    validation = df.iloc[train_end:validation_end].copy()
    test = df.iloc[validation_end:].copy()

    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("Uma das partições ficou vazia.")

    return SplitFrames(train=train, validation=validation, test=test)


def print_split_summary(split: SplitFrames, datetime_column: str) -> None:
    total = len(split.train) + len(split.validation) + len(split.test)
    for name, frame in (("TRAIN", split.train), ("VALIDATION", split.validation), ("TEST", split.test)):
        print(
            f"{name:<12} n={len(frame):>6} | "
            f"{len(frame) / total:>7.2%} | "
            f"{frame[datetime_column].min()} -> {frame[datetime_column].max()}"
        )


def save_split(split: SplitFrames, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    split.train.to_csv(output_dir / "train.csv", index=False)
    split.validation.to_csv(output_dir / "validation.csv", index=False)
    split.test.to_csv(output_dir / "test.csv", index=False)
