import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .config import (CSV_PATH, DATE_COL, TARGET_COL, LOOKBACK_days, LOOKBACK, EPOCHS, BATCH_SIZE,
                     MODELS_DIR, PLOTS_DIR)
from .data_io import load_data
from .features import build_data, refine_data
from .split import data_split
from .preprocess import fit_scaler, transform, sequences, save_scaler
from .models import build_model

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    df = load_data(str(CSV_PATH))
    refined_df = refine_data(df)                                 # return data with one row per day
    df = build_data(refined_df, TARGET_COL, LOOKBACK_days)       # build data with n number of dates of past

    X_train_df, y_train,  X_val_df,  y_val, X_test_df, y_test = data_split(df)      # split data into train, test, validation
    X_train_df, X_val_df, X_test_df, scalar = apply_scaling(train_df, val_df, test_df)  # scaling the X data
    save_model("scalar", scalar, MODELS_DIR)

    models = build_model(X_train,y_train, X_val, y_val)     # training the models
    for key in models:
        save_model(key, model[key], MODELS_DIR)             #   make a pkl file with the trained model
    model = models["best_model"]

    make_plots(model, X_train_df, y_train,  X_val_df,  y_val, X_test_df, y_test)

if __name__ == "__main__":
    main()
