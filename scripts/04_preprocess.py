from src.config import build_config
from src.data import SplitFrames, validate_columns
from src.preprocessing import fit_transform_splits, save_processed_splits
from pathlib import Path
import pandas as pd


cfg = build_config()

for scaler_name in cfg.scalers:
    for dataset in cfg.datasets:
        input_dir = cfg.split_dir / dataset
        train = pd.read_csv(input_dir / "train.csv")
        validation = pd.read_csv(input_dir / "validation.csv")
        test = pd.read_csv(input_dir / "test.csv")
        validate_columns(train, cfg.features)
        validate_columns(validation, cfg.features)
        validate_columns(test, cfg.features)

        transformed, scaler = fit_transform_splits(
            train=train,
            validation=validation,
            test=test,
            features=cfg.features,
            scaler_name=scaler_name,
            minmax_range=cfg.minmax_range,
        )

        output_dir = cfg.processed_dir / scaler_name / dataset
        save_processed_splits(transformed, scaler, output_dir, scaler_name)

        print(f"✓ {dataset} / {scaler_name} -> {output_dir}")
