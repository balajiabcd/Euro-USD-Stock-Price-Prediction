import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib, os
from .config import LOOKBACK

def fit_scaler(df: pd.DataFrame, cols: list[str]) -> MinMaxScaler:
    sc = MinMaxScaler()
    sc.fit(df[cols].astype(float))
    return sc

def transform(df: pd.DataFrame, cols: list[str], scaler: MinMaxScaler) -> np.ndarray:
    return scaler.transform(df[cols].astype(float))

def sequences(X: np.ndarray, y: np.ndarray, lookback: int = LOOKBACK):
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i-lookback:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

def save_scaler(scaler: MinMaxScaler, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(scaler, path)

def load_scaler(path: str) -> MinMaxScaler:
    return joblib.load(path)
