# src/evaluate.py
import os
import json
import numpy as np
import joblib

from keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .config import MODELS_DIR

def _load_pkl(name: str):
    """Load a pickle saved via save_object(name, item, MODELS_DIR)."""
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing artifact: {path}")
    return joblib.load(path)

def _load_best_model():
    """
    Load the best saved Keras model.
    Preference order: best_model.keras, model1.keras, best_model (SavedModel dir), model1 (SavedModel dir).
    """
    candidates = [
        os.path.join(MODELS_DIR, "best_model.keras"),
        os.path.join(MODELS_DIR, "model1.keras"),
        os.path.join(MODELS_DIR, "best_model"),
        os.path.join(MODELS_DIR, "model1"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return load_model(p)
    raise FileNotFoundError(
        "No saved model found in MODELS_DIR. "
        "Expected one of: best_model.keras, model1.keras, best_model/, model1/."
    )

def evaluate(save_metrics: bool = True, save_preds: bool = True, verbose: bool = True):
    """
    Evaluate the saved best model on the saved test split.
    - Loads X_test_df.pkl, y_test.pkl, (optionally scaler.pkl if needed later).
    - Predicts with the best model.
    - Computes MAE, RMSE, MAPE and writes metrics.json in MODELS_DIR.
    - Optionally saves y_pred_test.pkl for later plots.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Load artifacts (exact objects saved during training)
    X_test_df = _load_pkl("X_test_df")
    y_test    = _load_pkl("y_test")
    # scaler   = _load_pkl("scaler")  # Not needed for evaluation since X_test_df is already scaled

    # Load model
    model = _load_best_model()

    # Prepare arrays for LSTM: (N, timesteps, 1)
    X_test = X_test_df.values[..., np.newaxis]
    y_true = y_test.values if hasattr(y_test, "values") else np.asarray(y_test)

    # Predict
    y_pred = model.predict(X_test, verbose=0).ravel()

    # Metrics
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100.0)

    metrics = {"MAE": mae, "RMSE": rmse, "MAPE_percent": mape}

    if verbose:
        print(metrics)

    # Save metrics.json alongside your other pickles
    if save_metrics:
        with open(os.path.join(MODELS_DIR, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    # Save predictions for plotting later
    if save_preds:
        joblib.dump(y_pred, os.path.join(MODELS_DIR, "y_pred_test.pkl"))

    return metrics, y_true, y_pred

if __name__ == "__main__":
    evaluate()
