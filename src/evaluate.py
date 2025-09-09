import os, json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import load_model
from .config import CSV_PATH, TARGET_COL, LOOKBACK, MODELS_DIR
from .data_io import load_timeseries
from .features import add_technical_indicators
from .preprocess import load_scaler, transform, sequences

def evaluate():
    df = load_timeseries(str(CSV_PATH))
    df = add_technical_indicators(df, TARGET_COL)

    cfg = json.load(open(os.path.join(MODELS_DIR, "training_config.json")))
    feature_cols = cfg["feature_cols"]

    scaler = load_scaler(os.path.join(MODELS_DIR, "scaler.pkl"))
    X = transform(df, feature_cols, scaler)
    y = df[TARGET_COL].values.reshape(-1,1)

    Xs, ys = sequences(X, y, LOOKBACK)

    model = load_model(os.path.join(MODELS_DIR, "lstm_model.h5"))
    preds = model.predict(Xs, verbose=0)

    mae  = float(mean_absolute_error(ys, preds))
    rmse = float(np.sqrt(mean_squared_error(ys, preds)))
    mape = float(np.mean(np.abs((ys - preds) / np.clip(np.abs(ys), 1e-6, None))) * 100.0)

    out = {"MAE": mae, "RMSE": rmse, "MAPE": mape}
    with open(os.path.join(MODELS_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(out)

if __name__ == "__main__":
    evaluate()
