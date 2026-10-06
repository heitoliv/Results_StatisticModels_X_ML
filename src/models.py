from __future__ import annotations

from sklearn.neural_network import MLPRegressor
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout


def build_mlp(hidden_layers=(64, 32), max_iter=200, random_state=42):
    return MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        max_iter=max_iter,
        random_state=random_state,
        early_stopping=False,
    )


def build_lstm(lookback: int, n_features: int, units=64, dropout=0.2):
    model = Sequential([
        Input(shape=(lookback, n_features)),
        LSTM(units, return_sequences=False),
        Dropout(dropout),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model
