from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from tensorflow.keras.callbacks import EarlyStopping

from src.config import build_config
from src.metrics import calculate_metrics
from src.models import build_lstm, build_mlp
from src.preprocessing import inverse_transform_target
from src.reproducibility import configure_tensorflow_gpu, set_seed
from src.statistical_models import fit_and_forecast


def load_windows(base: Path, dataset: str, scaler: str):
    directory = base / scaler / dataset
    return {
        name: np.load(directory / f"{name}.npy")
        for name in (
            "X_train", "y_train", "X_validation", "y_validation",
            "X_test", "y_test", "validation_timestamps", "test_timestamps"
        )
    }


def save_prediction_file(directory: Path, model_name: str, split: str, timestamps, y_true_norm, y_pred_norm, y_true, y_pred):
    pd.DataFrame({
        "timestamp": timestamps,
        "y_true_normalized": y_true_norm,
        "y_pred_normalized": y_pred_norm,
        "y_true": y_true,
        "y_pred": y_pred,
    }).to_csv(directory / f"{model_name.lower()}_{split}_predictions.csv", index=False)


def evaluate_and_save(model_name, split, timestamps, y_true_norm, y_pred_norm, scaler, cfg, out_dir, scaler_name):
    y_true = inverse_transform_target(y_true_norm, scaler, cfg.features, cfg.target)
    y_pred = inverse_transform_target(y_pred_norm, scaler, cfg.features, cfg.target)
    metrics_original = calculate_metrics(y_true, y_pred)
    metrics_normalized = calculate_metrics(y_true_norm, y_pred_norm)

    save_prediction_file(out_dir, model_name, split, timestamps, y_true_norm, y_pred_norm, y_true, y_pred)

    return metrics_original, metrics_normalized


def evaluate_statistical_and_save(model_name, split, timestamps, y_true, y_pred, out_dir):
    """Avalia modelos estatísticos diretamente na escala original de OT."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    metrics_original = calculate_metrics(y_true, y_pred)
    # As colunas normalizadas são mantidas para compatibilidade do esquema,
    # mas não se aplicam a modelos ajustados nos dados brutos.
    save_prediction_file(
        out_dir, model_name, split, timestamps,
        np.full(len(y_true), np.nan), np.full(len(y_pred), np.nan), y_true, y_pred,
    )
    return metrics_original


def run_statistical_models(dataset: str, cfg) -> pd.DataFrame:
    """Executa os baselines estatísticos uma vez, sem qualquer scaler."""
    input_dir = cfg.split_dir / dataset
    frames = {
        split: pd.read_csv(input_dir / f"{split}.csv")
        for split in ("train", "validation", "test")
    }
    out_dir = cfg.baseline_dir / "raw" / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for model_name in ("ARIMA", "ARIMAX", "SARIMA", "SARIMAX"):
        print(f"Treinando {model_name} nos dados brutos...")
        forecast = fit_and_forecast(
            frames=frames,
            target=cfg.target,
            features=cfg.features,
            model_name=model_name,
            arima_order=cfg.arima_order,
            sarima_order=cfg.sarima_order,
            sarima_seasonal_order=cfg.sarima_seasonal_order,
            maxiter=cfg.stats_maxiter,
        )
        for split_name, prediction in (
            ("validation", forecast.validation),
            ("test", forecast.test),
        ):
            frame = frames[split_name]
            metrics_original = evaluate_statistical_and_save(
                model_name=model_name,
                split=split_name,
                timestamps=frame[cfg.datetime_column].astype(str).to_numpy(),
                y_true=frame[cfg.target].to_numpy(),
                y_pred=prediction,
                out_dir=out_dir,
            )
            rows.append({
                "scaler": "raw",
                "dataset": dataset,
                "model": model_name,
                "split": split_name,
                "train_seconds": forecast.train_seconds,
                **{f"{key}_original": value for key, value in metrics_original.items()},
                "MAE_normalized": np.nan,
                "RMSE_normalized": np.nan,
                "MAPE_normalized": np.nan,
                "R2_normalized": np.nan,
            })

    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "baseline_results.csv", index=False)
    return results


def run_one(scaler_name: str, dataset: str, cfg) -> pd.DataFrame:
    set_seed(cfg.seed)
    windows = load_windows(cfg.windows_dir, dataset, scaler_name)
    scaler = joblib.load(cfg.processed_dir / scaler_name / dataset / "scaler.pkl")

    out_dir = cfg.baseline_dir / scaler_name / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train = windows["X_train"]
    y_train = windows["y_train"]
    X_val = windows["X_validation"]
    y_val = windows["y_validation"]
    X_test = windows["X_test"]
    y_test = windows["y_test"]

    rows = []

    # ----------------------------- MLP ----------------------
    X_train_mlp = X_train.reshape(len(X_train), -1)
    X_val_mlp = X_val.reshape(len(X_val), -1)
    X_test_mlp = X_test.reshape(len(X_test), -1)

    mlp = build_mlp(cfg.mlp_hidden_layers, cfg.mlp_max_iter, cfg.mlp_random_state)
    start = time.perf_counter()
    mlp.fit(X_train_mlp, y_train)
    train_seconds = time.perf_counter() - start

    y_val_pred = mlp.predict(X_val_mlp)
    val_orig, val_norm = evaluate_and_save(
        "MLP", "validation", windows["validation_timestamps"], y_val, y_val_pred,
        scaler, cfg, out_dir, scaler_name
    )

    y_test_pred = mlp.predict(X_test_mlp)
    test_orig, test_norm = evaluate_and_save(
        "MLP", "test", windows["test_timestamps"], y_test, y_test_pred,
        scaler, cfg, out_dir, scaler_name
    )

    joblib.dump(mlp, out_dir / "mlp_model.pkl")

    for split_name, orig, norm in (("validation", val_orig, val_norm), ("test", test_orig, test_norm)):
        rows.append({
            "scaler": scaler_name,
            "dataset": dataset,
            "model": "MLP",
            "split": split_name,
            "train_seconds": train_seconds,
            **{f"{k}_original": v for k, v in orig.items()},
            **{f"{k}_normalized": v for k, v in norm.items()},
        })

    # ----------------------------- LSTM ---------------------
    set_seed(cfg.lstm_random_seed)
    lstm = build_lstm(cfg.lookback, X_train.shape[2], cfg.lstm_units, cfg.lstm_dropout)
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=cfg.lstm_patience,
        restore_best_weights=True,
        mode="min",
    )

    start = time.perf_counter()
    history = lstm.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=cfg.lstm_epochs,
        batch_size=cfg.lstm_batch_size,
        callbacks=[early_stopping],
        shuffle=False,
        verbose=1,
    )
    train_seconds = time.perf_counter() - start

    pd.DataFrame(history.history).to_csv(out_dir / "lstm_training_history.csv", index=False)

    y_val_pred = lstm.predict(X_val, verbose=0).reshape(-1)
    val_orig, val_norm = evaluate_and_save(
        "LSTM", "validation", windows["validation_timestamps"], y_val, y_val_pred,
        scaler, cfg, out_dir, scaler_name
    )

    y_test_pred = lstm.predict(X_test, verbose=0).reshape(-1)
    test_orig, test_norm = evaluate_and_save(
        "LSTM", "test", windows["test_timestamps"], y_test, y_test_pred,
        scaler, cfg, out_dir, scaler_name
    )

    lstm.save(out_dir / "lstm_model.keras")

    for split_name, orig, norm in (("validation", val_orig, val_norm), ("test", test_orig, test_norm)):
        rows.append({
            "scaler": scaler_name,
            "dataset": dataset,
            "model": "LSTM",
            "split": split_name,
            "train_seconds": train_seconds,
            **{f"{k}_original": v for k, v in orig.items()},
            **{f"{k}_normalized": v for k, v in norm.items()},
        })

    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "baseline_results.csv", index=False)
    return results


def main():
    cfg = build_config()
    configure_tensorflow_gpu()
    all_results = []

    for scaler_name in cfg.scalers:
        for dataset in cfg.datasets:
            print("\n" + "=" * 80)
            print(f"BASELINE | scaler={scaler_name} | dataset={dataset}")
            print("=" * 80)
            results = run_one(scaler_name, dataset, cfg)
            all_results.append(results)

    for dataset in cfg.datasets:
        print("\n" + "=" * 80)
        print(f"BASELINES ESTATÍSTICOS (dados brutos) | dataset={dataset}")
        print("=" * 80)
        all_results.append(run_statistical_models(dataset, cfg))

    final = pd.concat(all_results, ignore_index=True)
    cfg.baseline_dir.mkdir(parents=True, exist_ok=True)
    final.to_csv(cfg.baseline_dir / "baseline_all_results.csv", index=False)

    # Comparação de preprocessamento: somente redes neurais recebem scaler.
    test = final[final["split"] == "test"].copy()
    neural_test = test[test["model"].isin(["MLP", "LSTM"])]
    pivot = neural_test.pivot_table(
        index=["dataset", "model"],
        columns="scaler",
        values=["RMSE_original", "MAE_original", "MAPE_original", "R2_original"],
    )
    pivot.to_csv(cfg.baseline_dir / "baseline_scaler_comparison_test.csv")

    statistical_test = test[test["model"].isin(["ARIMA", "ARIMAX", "SARIMA", "SARIMAX"])]
    statistical_test.to_csv(cfg.baseline_dir / "statistical_models_comparison_test.csv", index=False)

    print("\nResultados finais:\n")
    print(test[["scaler", "dataset", "model", "RMSE_original", "MAE_original", "MAPE_original", "R2_original"]].to_string(index=False))


if __name__ == "__main__":
    main()
