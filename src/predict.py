# src/predict.py
import os, json
import numpy as np, pandas as pd, joblib
from keras.models import load_model
from .config import MODELS_DIR, LOOKBACK_days

def predict_rate(x,y):
    model = load_model(os.path.join(MODELS_DIR, "best_model.keras"))
    X = x.values.reshape(-1, LOOKBACK_days, 1)
    
    y_pred = model.predict(x).ravel()
    y_true = np.asarray(getattr(y, "values", y)).ravel()
    for i in range(len(y_true)):
        print(f"original is {y_true[i]:.4f}, prediction is {y_pred[i]:.4f}")


if __name__ == "__main__":
    X_test_df = joblib.load(os.path.join(MODELS_DIR, "X_test_df.pkl"))
    y_test = joblib.load(os.path.join(MODELS_DIR, "y_test.pkl"))
    x = X_test_df.tail(5)
    y = y_test.iloc[-5:] 
    predict_rate(x,y)
