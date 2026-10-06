from src.config import build_config
from src.data import validate_columns
from src.windowing import create_windows_by_target_split, validate_windows, save_windows
import pandas as pd


cfg = build_config()

for scaler_name in cfg.scalers:
    for dataset in cfg.datasets:
        input_dir = cfg.processed_dir / scaler_name / dataset
        frames = {
            "train": pd.read_csv(input_dir / "train.csv"),
            "validation": pd.read_csv(input_dir / "validation.csv"),
            "test": pd.read_csv(input_dir / "test.csv"),
        }
        for frame in frames.values():
            validate_columns(frame, cfg.features)

        windows = create_windows_by_target_split(
            scaled_frames=frames,
            features=cfg.features,
            target=cfg.target,
            datetime_column=cfg.datetime_column,
            lookback=cfg.lookback,
            horizon=cfg.horizon,
        )
        validate_windows(windows)
        output_dir = cfg.windows_dir / scaler_name / dataset
        save_windows(windows, output_dir)
        print(
            f"✓ {dataset} / {scaler_name}: "
            f"train={windows.X_train.shape}, "
            f"val={windows.X_validation.shape}, "
            f"test={windows.X_test.shape}"
        )
