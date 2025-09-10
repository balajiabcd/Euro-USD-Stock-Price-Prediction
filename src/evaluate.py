# src/evaluate.py
import os
import json
import numpy as np
import joblib

from keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .config import MODELS_DIR, LOOKBACK_days



def load_pkl(name: str):
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing artifact: {path}")
    return joblib.load(path)



def load_best_model():
    return load_model(os.path.join(MODELS_DIR, "best_model.keras"))



def evaluate(save_metrics: bool = True, save_preds: bool = True, verbose: bool = True):

    X_test_df = load_pkl("X_test_df")
    y_test    = load_pkl("y_test")
    model = load_best_model()

    X_test = X_test_df.values.reshape(-1, LOOKBACK_days, 1)
    y_true = np.asarray(getattr(y_test, "values", y_test)).ravel()
    y_pred = model.predict(X_test).ravel()


    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100.0)

    metrics = {"MAE": mae, "RMSE": rmse, "MAPE_percent": mape}
    print(metrics)
    
    with open(os.path.join(MODELS_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if save_preds:
        joblib.dump(y_pred, os.path.join(MODELS_DIR, "y_pred_test.pkl"))
        joblib.dump(y_true, os.path.join(MODELS_DIR, "y_true_test.pkl"))
        
    return metrics, y_true, y_pred

if __name__ == "__main__":
    evaluate()
