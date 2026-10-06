"""Baselines ARIMA/SARIMA com previsão recursiva de um passo à frente."""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


@dataclass(frozen=True)
class StatisticalForecasts:
    validation: np.ndarray
    test: np.ndarray
    train_seconds: float


def _one_step_predictions(result, values, exog, start: int, end: int) -> np.ndarray:
    """Prevê t usando apenas informação até t-1 e atualiza o estado com y(t)."""
    predictions = []
    for index in range(start, end):
        exog_row = None if exog is None else exog[index - 1:index]
        if exog_row is None:
            prediction = result.forecast(steps=1)
            # extend mantém apenas o estado de filtragem mais recente; é muito
            # mais econômico que append para milhares de previsões sequenciais.
            result = result.extend([values[index]])
        else:
            prediction = result.forecast(steps=1, exog=exog_row)
            result = result.extend([values[index]], exog=exog_row)
        predictions.append(float(np.asarray(prediction).reshape(-1)[0]))
    return np.asarray(predictions, dtype=np.float64)


def fit_and_forecast(
    frames: dict[str, pd.DataFrame],
    target: str,
    features: tuple[str, ...],
    model_name: str,
    arima_order: tuple[int, int, int],
    sarima_order: tuple[int, int, int],
    sarima_seasonal_order: tuple[int, int, int, int],
    maxiter: int,
) -> StatisticalForecasts:
    """Ajusta no TRAIN e avalia validation/test sem vazamento de futuro.

    ARIMAX/SARIMAX recebem covariáveis defasadas em uma amostra. Portanto, a
    previsão para t só usa as variáveis que já estavam disponíveis em t-1,
    assim como as janelas da MLP/LSTM deste projeto.
    """
    full = pd.concat(
        [frames["train"], frames["validation"], frames["test"]],
        ignore_index=True,
    )
    values = full[target].to_numpy(dtype=np.float64)
    train_end = len(frames["train"])
    validation_end = train_end + len(frames["validation"])
    if train_end < 3:
        raise ValueError("TRAIN precisa de ao menos três observações para os modelos estatísticos.")

    uses_exog = model_name in {"ARIMAX", "SARIMAX"}
    exog_columns = [column for column in features if column != target]
    exog = full.loc[:, exog_columns].to_numpy(dtype=np.float64) if uses_exog else None
    # A primeira observação não tem covariáveis defasadas disponíveis.
    fit_endog = values[1:train_end]
    fit_exog = None if exog is None else exog[:train_end - 1]

    if model_name == "ARIMA":
        model = ARIMA(fit_endog, order=arima_order, exog=fit_exog)
    elif model_name == "ARIMAX":
        # SARIMAX sem parte sazonal é a formulação state-space de ARIMAX e
        # permite atualizar as covariáveis defasadas eficientemente com extend.
        model = SARIMAX(
            fit_endog,
            order=arima_order,
            exog=fit_exog,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
    elif model_name in {"SARIMA", "SARIMAX"}:
        model = SARIMAX(
            fit_endog,
            order=sarima_order,
            seasonal_order=sarima_seasonal_order,
            exog=fit_exog,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
    else:
        raise ValueError(f"Modelo estatístico desconhecido: {model_name}")

    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        if model_name == "ARIMA":
            result = model.fit(method_kwargs={"maxiter": maxiter})
        else:
            result = model.fit(maxiter=maxiter, disp=False)
    train_seconds = time.perf_counter() - start

    validation = _one_step_predictions(result, values, exog, train_end, validation_end)
    # O TEST usa o estado ajustado no TRAIN e os valores realmente observados
    # na validação; nunca usa suas próprias previsões como entrada.
    result_for_test = result
    for index in range(train_end, validation_end):
        exog_row = None if exog is None else exog[index - 1:index]
        if exog_row is None:
            result_for_test = result_for_test.extend([values[index]])
        else:
            result_for_test = result_for_test.extend([values[index]], exog=exog_row)
    test = _one_step_predictions(result_for_test, values, exog, validation_end, len(full))
    return StatisticalForecasts(validation=validation, test=test, train_seconds=train_seconds)
