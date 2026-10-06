from src.config import build_config
from src.data import load_dataset, validate_columns
from src.plotting import plot_raw_series


cfg = build_config()
output_dir = cfg.project_root / "image_series"

for dataset in cfg.datasets:
    df = load_dataset(cfg.raw_data_dir / f"{dataset}.csv", cfg.datetime_column)
    validate_columns(df, cfg.features)

    print("\n" + "=" * 70)
    print(f"EDA: {dataset}")
    print("=" * 70)
    print(f"Dimensões: {df.shape}")
    print(f"Período: {df[cfg.datetime_column].min()} -> {df[cfg.datetime_column].max()}")
    print(f"Nulos: {df.isna().sum().sum()}")
    print(f"Linhas duplicadas: {df.duplicated().sum()}")
    print(f"Datas duplicadas: {df[cfg.datetime_column].duplicated().sum()}")
    print("\nIntervalos observados:")
    print(df[cfg.datetime_column].diff().dropna().value_counts().head())
    print("\nEstatística descritiva:")
    print(df[list(cfg.features)].describe().T.round(3))

    plot_raw_series(df, dataset, cfg.features, cfg.datetime_column, output_dir)
    print(f"✓ Gráficos salvos em {output_dir}")
