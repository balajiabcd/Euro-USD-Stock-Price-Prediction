import os, json
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from .config import CSV_PATH, TARGET_COL, LOOKBACK, MODELS_DIR, DATE_COL
from .data_io import load_timeseries
from .features import add_technical_indicators
from .preprocess import load_scaler, transform

def predict_next(n_steps: int = 1):
    df = load_timeseries(str(CSV_PATH))
    df = add_technical_indicators(df, TARGET_COL)

    cfg = json.load(open(os.path.join(MODELS_DIR, "training_config.json")))
    feature_cols = cfg["feature_cols"]
    scaler = load_scaler(os.path.join(MODELS_DIR, "scaler.pkl"))

    X = transform(df, feature_cols, scaler)
    model = load_model(os.path.join(MODELS_DIR, "lstm_model.h5"))

    window = X[-LOOKBACK:]
    preds = []
    for _ in range(n_steps):
        y = float(model.predict(window[np.newaxis, ...], verbose=0)[0,0])
        preds.append(y)
        try:
            idx = feature_cols.index(TARGET_COL)
            next_row = window[-1].copy()
            next_row[idx] = y
        except ValueError:
            next_row = window[-1].copy()
        window = np.vstack([window[1:], next_row])

    if DATE_COL in df.columns:
        last_date = pd.to_datetime(df[DATE_COL].iloc[-1])
        dates = [str(last_date + pd.Timedelta(days=i+1)) for i in range(n_steps)]
    else:
        dates = list(range(len(df), len(df)+n_steps))

    return list(zip(dates, preds))

if __name__ == "__main__":
    for d, y in predict_next(5):
        print(d, y)
