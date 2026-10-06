from src.config import build_config
from src.plotting import plot_model_comparison, plot_predictions
import pandas as pd


cfg = build_config()
results_file = cfg.baseline_dir / "baseline_all_results.csv"
if not results_file.exists():
    raise FileNotFoundError(f"Resultado não encontrado: {results_file}")

results = pd.read_csv(results_file)
output_dir = cfg.baseline_dir / "plots"
output_dir.mkdir(parents=True, exist_ok=True)

for metric in ("RMSE_original", "MAE_original", "MAPE_original", "R2_original"):
    plot_model_comparison(results, output_dir / f"comparison_{metric.lower()}.png", metric)

# Consolida as previsões na escala original e gera um gráfico de comparação
# real x previsto para cada modelo, scaler, dataset e partição.
prediction_dir = output_dir / "predictions"
prediction_dir.mkdir(parents=True, exist_ok=True)
all_predictions = []
for row in results.itertuples(index=False):
    prediction_file = (
        cfg.baseline_dir / row.scaler / row.dataset /
        f"{row.model.lower()}_{row.split}_predictions.csv"
    )
    if not prediction_file.exists():
        print(f"Aviso: previsão não encontrada: {prediction_file}")
        continue

    prediction = pd.read_csv(prediction_file)
    prediction.insert(0, "split", row.split)
    prediction.insert(0, "model", row.model)
    prediction.insert(0, "dataset", row.dataset)
    prediction.insert(0, "scaler", row.scaler)
    all_predictions.append(prediction)

    plot_predictions(
        y_true=prediction["y_true"].to_numpy(),
        y_pred=prediction["y_pred"].to_numpy(),
        timestamps=prediction["timestamp"].to_numpy(),
        title=f"{row.dataset} | {row.scaler} | {row.model} | {row.split}",
        output_file=prediction_dir / f"{row.scaler}_{row.dataset}_{row.model.lower()}_{row.split}.png",
    )
    if row.split == "test":
        plot_predictions(
            y_true=prediction["y_true"].to_numpy(),
            y_pred=prediction["y_pred"].to_numpy(),
            timestamps=prediction["timestamp"].to_numpy(),
            title=(
                f"{row.dataset} | {row.scaler} | {row.model} | TEST "
                f"(zoom: últimos {cfg.prediction_zoom_points} pontos)"
            ),
            output_file=(
                prediction_dir /
                f"{row.scaler}_{row.dataset}_{row.model.lower()}_test_zoom.png"
            ),
            limit=cfg.prediction_zoom_points,
            tail=True,
        )

if all_predictions:
    pd.concat(all_predictions, ignore_index=True).to_csv(
        cfg.baseline_dir / "baseline_all_predictions.csv", index=False
    )

# Comparação específica de preprocessamentos no TEST: somente redes neurais.
test = results[results["split"] == "test"].copy()
neural_test = test[test["model"].isin(["MLP", "LSTM"])]
comparison = neural_test.pivot_table(
    index=["dataset", "model"],
    columns="scaler",
    values=["RMSE_original", "MAE_original", "MAPE_original", "R2_original"],
)
comparison.to_csv(cfg.baseline_dir / "preprocessing_comparison_test.csv")

statistical_test = test[test["model"].isin(["ARIMA", "ARIMAX", "SARIMA", "SARIMAX"])]
statistical_test.to_csv(cfg.baseline_dir / "statistical_models_comparison_test.csv", index=False)

print(comparison)
print(f"\n✓ Métricas e gráficos salvos em {output_dir}")
print(f"✓ Previsões consolidadas: {cfg.baseline_dir / 'baseline_all_predictions.csv'}")
