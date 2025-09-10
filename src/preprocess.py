import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib, os



def make_fit_scaler(df: pd.DataFrame, cols: list[str]):
    sc = MinMaxScaler()
    sc.fit(df[cols].astype(float))
    return sc



def transform(df: pd.DataFrame, cols: list[str], scaler: MinMaxScaler):
    scaled_df = pd.DataFrame(   scaler.transform(df[cols].astype(float)),
                                columns=cols, index=df.index)
    return scaled_df 



def save_object(name, item, path):
    os.makedirs(path, exist_ok=True)
    joblib.dump(item, os.path.join(path, f"{name}.pkl"))

