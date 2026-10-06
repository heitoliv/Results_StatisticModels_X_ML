from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_raw_series(df: pd.DataFrame, dataset_name: str, variables: tuple[str, ...], datetime_column: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for variable in variables:
        fig, ax = plt.subplots(figsize=(16, 6))
        ax.plot(df[datetime_column], df[variable], linewidth=0.8)
        ax.set_title(f"{dataset_name} — Série temporal de {variable}")
        ax.set_xlabel("Data")
        ax.set_ylabel(variable)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / f"{dataset_name}_{variable}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_predictions(
    y_true,
    y_pred,
    title: str,
    output_file: Path,
    timestamps=None,
    limit: int | None = None,
    tail: bool = False,
) -> None:
    """Plota previsão e valor original; por padrão preserva toda a série."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    timestamps = None if timestamps is None else np.asarray(timestamps)
    n = len(y_true) if limit is None else min(limit, len(y_true))
    start = len(y_true) - n if tail else 0
    stop = start + n
    fig, ax = plt.subplots(figsize=(16, 6))
    x_axis = np.arange(n) if timestamps is None else pd.to_datetime(timestamps[start:stop])
    ax.plot(x_axis, y_true[start:stop], label="Valor original (OT)", linewidth=1.3)
    ax.plot(x_axis, y_pred[start:stop], label="Previsão", linewidth=1.0, alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("Data" if timestamps is not None else "Índice temporal")
    ax.set_ylabel("OT")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if timestamps is not None:
        fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(df: pd.DataFrame, output_file: Path, metric: str = "RMSE_original") -> None:
    """Comparação por dataset/model/scaler no TEST."""
    test = df[df["split"] == "test"].copy()
    if test.empty:
        return

    groups = test[["dataset", "model", "scaler", metric]].copy()
    pivot = groups.pivot_table(index="dataset", columns=["model", "scaler"], values=metric)

    fig, ax = plt.subplots(figsize=(12, 6))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title(f"Comparação no TEST — {metric}")
    ax.set_xlabel("Dataset")
    ax.set_ylabel(metric)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
