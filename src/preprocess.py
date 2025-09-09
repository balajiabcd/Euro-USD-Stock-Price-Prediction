import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib, os
from .config import LOOKBACK

def scalling(df1: pd.DataFrame, df2: pd.DataFrame, cols_train: list[str], cols_test: list[str]):
    sc = fit_scaler(df: pd.DataFrame, cols: list[str])
    X_train = transform(df: pd.DataFrame, cols: list[str], scaler: MinMaxScaler)
    X_test = transform(df: pd.DataFrame, cols: list[str], scaler: MinMaxScaler)
    X_val = transform(df: pd.DataFrame, cols: list[str], scaler: MinMaxScaler)

def fit_scaler(df: pd.DataFrame, cols: list[str]) -> MinMaxScaler:
    sc = MinMaxScaler()
    sc.fit(df[cols].astype(float))
    return sc

def transform(df: pd.DataFrame, cols: list[str], scaler: MinMaxScaler) -> np.ndarray:
    return scaler.transform(df[cols].astype(float))

def save_scaler(scaler: MinMaxScaler, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(scaler, path)





def sequences(X: np.ndarray, y: np.ndarray, lookback: int = LOOKBACK):
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i-lookback:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

def load_scaler(path: str) -> MinMaxScaler:
    return joblib.load(path)



 os.path.join(MODELS_DIR, "scaler.pkl")