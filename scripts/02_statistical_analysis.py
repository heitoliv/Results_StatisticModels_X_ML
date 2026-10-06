from src.config import build_config
from src.data import load_dataset, split_chronologically, validate_columns
from src.statistics_analysis import run_target_analysis


cfg = build_config()

for dataset in cfg.datasets:
    df = load_dataset(cfg.raw_data_dir / f"{dataset}.csv", cfg.datetime_column)
    validate_columns(df, cfg.features)
    split = split_chronologically(df, cfg.train_ratio, cfg.validation_ratio, cfg.test_ratio)

    # Para testes estatísticos que possam influenciar decisões do modelo,
    # usamos somente TRAIN e evitamos contaminar a análise com o TEST.
    output_dir = cfg.stats_dir / dataset
    run_target_analysis(split.train, cfg.target, output_dir)
    print(f"✓ Análise estatística salva em: {output_dir}")
