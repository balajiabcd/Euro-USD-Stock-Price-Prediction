import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .config import (CSV_PATH, TARGET_COL, LOOKBACK_days, EPOCHS, MODELS_DIR)
from .data_io import load_data
from .features import build_data, refine_data
from .split import data_split
from .preprocess import save_object, make_fit_scaler, transform
from .models import build_models


 
def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    df = load_data(str(CSV_PATH))
    refined_df = refine_data(df)                                 # return data with one row per day
    df = build_data(refined_df, TARGET_COL, LOOKBACK_days)       # build data with n number of dates of past

    X_train_df, y_train,  X_val_df,  y_val, X_test_df, y_test = data_split(df)          # split data into train, test, validation
    
    scaler = make_fit_scaler(X_train_df, X_train_df.columns)
    X_train_df = pd.DataFrame(transform(X_train_df, X_train_df.columns, scaler),
                                        columns=X_train_df.columns, index=X_train_df.index)
    X_val_df   = pd.DataFrame(transform(X_val_df,   X_val_df.columns,   scaler),
                                        columns=X_val_df.columns,   index=X_val_df.index)
    X_test_df  = pd.DataFrame(transform(X_test_df,  X_test_df.columns,  scaler),
                                        columns=X_test_df.columns,  index=X_test_df.index)


    X_train = X_train_df.to_numpy().reshape(-1, LOOKBACK_days, 1)
    X_val   = X_val_df.to_numpy().reshape(-1, LOOKBACK_days, 1)
    X_test  = X_test_df.to_numpy().reshape(-1, LOOKBACK_days, 1)

    
    data_dict = {   "X_train_df": X_train_df,   "y_train": y_train,
                    "X_val_df": X_val_df,       "y_val": y_val,
                    "X_test_df": X_test_df,     "y_test": y_test,   "scaler": scaler }

    for key, value in data_dict.items():
        save_object(key, value, MODELS_DIR)

    models = build_models(X_train, y_train, X_val, y_val, EPOCHS)       # training models
    for key, value in models.items():
        value.save(os.path.join(MODELS_DIR, f"{key}.keras"))                        # not pkl file, using kras library save to save neural network models
    model = models["best_model"]



if __name__ == "__main__":
    main()
