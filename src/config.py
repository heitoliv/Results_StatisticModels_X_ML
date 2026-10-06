from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class Config:
    """Configuração central do experimento de baseline."""

    project_root: Path
    datasets: Tuple[str, ...] = ("ETTh1", "ETTh2")
    datetime_column: str = "date"
    features: Tuple[str, ...] = (
        "HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"
    )
    target: str = "OT"

    train_ratio: float = 0.70
    validation_ratio: float = 0.10
    test_ratio: float = 0.20

    scalers: Tuple[str, ...] = ("minmax", "standard")
    minmax_range: Tuple[float, float] = (0.0, 1.0)

    lookback: int = 24
    horizon: int = 1
    prediction_zoom_points: int = 168  # uma semana para dados horários

    seed: int = 42

    mlp_hidden_layers: Tuple[int, ...] = (64, 32)
    mlp_max_iter: int = 200
    mlp_random_state: int = 42

    lstm_units: int = 64
    lstm_dropout: float = 0.20
    lstm_epochs: int = 100
    lstm_batch_size: int = 32
    lstm_patience: int = 10
    lstm_random_seed: int = 42

    # Modelos estatísticos. As ordens foram mantidas deliberadamente simples
    # para que sirvam como baselines reproduzíveis; ajuste-as com base na ACF,
    # PACF e nos testes salvos em data/statistics antes do experimento final.
    arima_order: Tuple[int, int, int] = (1, 1, 1)
    sarima_order: Tuple[int, int, int] = (1, 1, 1)
    sarima_seasonal_order: Tuple[int, int, int, int] = (1, 0, 1, 24)
    stats_maxiter: int = 100

    @property
    def raw_data_dir(self) -> Path:
        return self.project_root / "data" / "ETDataset-main" / "ETT-small"

    @property
    def split_dir(self) -> Path:
        return self.project_root / "data" / "split"

    @property
    def processed_dir(self) -> Path:
        return self.project_root / "data" / "processed"

    @property
    def windows_dir(self) -> Path:
        return self.project_root / "data" / "windows"

    @property
    def baseline_dir(self) -> Path:
        return self.project_root / "data" / "baseline"

    @property
    def stats_dir(self) -> Path:
        return self.project_root / "data" / "statistics"

    def validate(self) -> None:
        total = self.train_ratio + self.validation_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-12:
            raise ValueError(f"As proporções devem somar 1.0. Obtido: {total}")
        if self.target not in self.features:
            raise ValueError("O target deve estar presente nas features quando o histórico do target é usado como entrada.")
        if self.lookback < 1 or self.horizon < 1:
            raise ValueError("lookback e horizon devem ser >= 1.")
        invalid_scalers = set(self.scalers) - {"minmax", "standard"}
        if invalid_scalers:
            raise ValueError(f"Scalers inválidos: {sorted(invalid_scalers)}")


def build_config(project_root: Path | None = None) -> Config:
    # config.py fica em <projeto>/src; portanto parents[1] é a raiz do projeto.
    root = project_root or Path(__file__).resolve().parents[1]
    config = Config(project_root=root)
    config.validate()
    return config
