from pathlib import Path

import pandas as pd

from src.config import build_config
from src.data import load_dataset, validate_columns


cfg = build_config()

for dataset in cfg.datasets:
    path = cfg.raw_data_dir / f"{dataset}.csv"
    df = load_dataset(path, cfg.datetime_column)
    validate_columns(df, cfg.features)

    print("\n" + "=" * 70)
    print(f"DATASET: {dataset}")
    print("=" * 70)
    print(f"Dimensões: {df.shape}")
    print(f"Período: {df[cfg.datetime_column].min()} -> {df[cfg.datetime_column].max()}")
    print(f"Nulos totais: {df.isna().sum().sum()}")
    print(f"Duplicadas: {df.duplicated().sum()}")
    print(f"Datas duplicadas: {df[cfg.datetime_column].duplicated().sum()}")
    print(df[list(cfg.features)].describe().T.round(4))
