from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.seasonal import STL


def descriptive_statistics(df: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    desc = df[numeric_columns].describe().T
    desc["median"] = df[numeric_columns].median()
    desc["skewness"] = df[numeric_columns].skew()
    desc["kurtosis"] = df[numeric_columns].kurtosis()
    desc["missing"] = df[numeric_columns].isna().sum()
    return desc.reset_index(names="variable")


def stationarity_report(series: pd.Series, name: str) -> dict:
    x = series.dropna().astype(float).to_numpy()

    adf_stat, adf_p, _, _, _, _ = adfuller(x, autolag="AIC")
    try:
        kpss_stat, kpss_p, _, _ = kpss(x, regression="c", nlags="auto")
    except Exception:
        kpss_stat, kpss_p = np.nan, np.nan

    return {
        "series": name,
        "adf_statistic": adf_stat,
        "adf_pvalue": adf_p,
        "kpss_statistic": kpss_stat,
        "kpss_pvalue": kpss_p,
        "adf_stationary_5pct": adf_p < 0.05,
        "kpss_stationary_5pct": kpss_p > 0.05 if np.isfinite(kpss_p) else np.nan,
    }


def trend_report(series: pd.Series, name: str) -> dict:
    y = series.dropna().astype(float).to_numpy()
    x = np.arange(len(y))
    result = linregress(x, y)
    return {
        "series": name,
        "slope_per_observation": result.slope,
        "pvalue": result.pvalue,
        "rvalue": result.rvalue,
        "significant_trend_5pct": result.pvalue < 0.05,
    }


def autocorrelation_report(series: pd.Series, name: str, lags=(1, 24, 48, 168)) -> pd.DataFrame:
    x = series.dropna().astype(float).to_numpy()
    max_lag = max(lags)
    values = acf(x, nlags=max_lag, fft=True)
    return pd.DataFrame({
        "series": name,
        "lag": list(lags),
        "acf": [values[lag] for lag in lags],
    })


def ljung_box_report(series: pd.Series, name: str, lags=(24, 48, 168)) -> pd.DataFrame:
    x = series.dropna().astype(float)
    result = acorr_ljungbox(x, lags=list(lags), return_df=True)
    result = result.reset_index(names="lag")
    result.insert(0, "series", name)
    return result


def save_acf_pacf(series: pd.Series, name: str, output_dir: Path, max_lag: int = 168) -> None:
    x = series.dropna().astype(float).to_numpy()
    acf_values = acf(x, nlags=max_lag, fft=True)
    pacf_values = pacf(x, nlags=max_lag, method="ywm")

    for metric_name, values in (("acf", acf_values), ("pacf", pacf_values)):
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.stem(range(len(values)), values)
        ax.set_title(f"{name} — {metric_name.upper()}")
        ax.set_xlabel("Lag")
        ax.set_ylabel(metric_name.upper())
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / f"{name}_{metric_name}.png", dpi=250)
        plt.close(fig)


def save_stl(series: pd.Series, name: str, output_dir: Path, period: int = 24) -> None:
    x = series.dropna().astype(float)
    result = STL(x, period=period, robust=True).fit()
    result_df = pd.DataFrame({
        "observed": result.observed,
        "trend": result.trend,
        "seasonal": result.seasonal,
        "resid": result.resid,
    })
    result_df.to_csv(output_dir / f"{name}_stl_period_{period}.csv", index=False)

    fig = result.plot()
    fig.set_size_inches(14, 9)
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}_stl_period_{period}.png", dpi=250)
    plt.close(fig)



def correlation_matrix(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return df[columns].corr(method="pearson")


def seasonal_lag_correlations(series: pd.Series, lags=(24, 168)) -> pd.DataFrame:
    x = series.astype(float)
    rows = []
    for lag in lags:
        if lag >= len(x):
            continue
        rows.append({"lag": lag, "correlation": x.autocorr(lag=lag)})
    return pd.DataFrame(rows)


def stationarity_after_differencing(series: pd.Series, name: str, order: int = 1) -> dict:
    diff = series.astype(float).diff(order).dropna()
    report = stationarity_report(diff, name=f"{name}_diff_{order}")
    return report

def run_target_analysis(train_df: pd.DataFrame, target: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    series = train_df[target]

    descriptive_statistics(train_df, [target]).to_csv(output_dir / "descriptive_statistics.csv", index=False)
    pd.DataFrame([stationarity_report(series, target)]).to_csv(output_dir / "stationarity.csv", index=False)
    pd.DataFrame([trend_report(series, target)]).to_csv(output_dir / "trend.csv", index=False)
    autocorrelation_report(series, target).to_csv(output_dir / "acf_selected_lags.csv", index=False)
    ljung_box_report(series, target).to_csv(output_dir / "ljung_box.csv", index=False)
    save_acf_pacf(series, target, output_dir)
    save_stl(series, target, output_dir, period=24)
    save_stl(series, target, output_dir, period=168)
    seasonal_lag_correlations(series).to_csv(output_dir / "seasonal_lag_correlations.csv", index=False)
    pd.DataFrame([stationarity_after_differencing(series, target, order=1)]).to_csv(output_dir / "stationarity_first_difference.csv", index=False)
