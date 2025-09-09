import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .config import (CSV_PATH, DATE_COL, TARGET_COL, LOOKBACK, EPOCHS, BATCH_SIZE,
                     MODELS_DIR, PLOTS_DIR)
from .data_io import load_timeseries
from .features import add_technical_indicators
from .split import temporal_split
from .preprocess import fit_scaler, transform, sequences, save_scaler
from .models import build_model

def plot_history(history, path):
    plt.figure()
    plt.plot(history.history.get("loss", []), label="train_loss")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="val_loss")
    plt.legend(); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.tight_layout()
    plt.savefig(path); plt.close()

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    df = load_timeseries(str(CSV_PATH))
    df = add_technical_indicators(df, TARGET_COL)

    train_df, val_df, test_df = temporal_split(df)
    feature_cols = df.select_dtypes(include=["number"]).columns.tolist()
    assert TARGET_COL in feature_cols, f"TARGET_COL '{TARGET_COL}' must be numeric"

    scaler = fit_scaler(train_df, feature_cols)
    Xtr = transform(train_df, feature_cols, scaler); ytr = train_df[TARGET_COL].values.reshape(-1,1)
    Xv  = transform(val_df, feature_cols, scaler);   yv  = val_df[TARGET_COL].values.reshape(-1,1)

    X_tr, y_tr = sequences(Xtr, ytr, LOOKBACK)
    X_va, y_va = sequences(Xv,  yv,  LOOKBACK)

    model = build_model(input_shape=(X_tr.shape[1], X_tr.shape[2]))
    history = model.fit(X_tr, y_tr, validation_data=(X_va, y_va) if len(X_va)>0 else None,
                        epochs=40, batch_size=32, verbose=2)

    model_path = os.path.join(MODELS_DIR, "lstm_model.h5")
    model.save(model_path)
    save_scaler(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    with open(os.path.join(MODELS_DIR, "training_config.json"), "w", encoding="utf-8") as f:
        json.dump({"feature_cols": feature_cols, "LOOKBACK": 5}, f, indent=2)

    plot_history(history, os.path.join(PLOTS_DIR, "training_history.png"))

if __name__ == "__main__":
    main()
