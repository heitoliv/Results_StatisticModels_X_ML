from src.config import build_config
from src.data import load_dataset, split_chronologically, print_split_summary, save_split, validate_columns


cfg = build_config()

for dataset in cfg.datasets:
    df = load_dataset(cfg.raw_data_dir / f"{dataset}.csv", cfg.datetime_column)
    validate_columns(df, cfg.features)
    split = split_chronologically(df, cfg.train_ratio, cfg.validation_ratio, cfg.test_ratio)
    output_dir = cfg.split_dir / dataset
    save_split(split, output_dir)

    print("\n" + "=" * 70)
    print(f"SPLIT: {dataset}")
    print("=" * 70)
    print_split_summary(split, cfg.datetime_column)
