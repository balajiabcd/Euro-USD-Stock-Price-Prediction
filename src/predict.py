# src/predict.py
import os, json
import numpy as np, pandas as pd, joblib
from keras.models import load_model
from .config import MODELS_DIR

def predict_rate(x,y):
    model = load_model(os.path.join(MODELS_DIR, "best_model.keras"))
    
    y_pred = model.predict(x)
    for i in range(len(x)):
        y1, y2 = round(y_test.iloc[-i],4), round(float(y_pred[i]),4)
        print(f"original is {y1}, prediciton is {y2}")


if __name__ == "__main__":
    X_test_df = joblib.load(os.path.join(MODELS_DIR, "X_test_df.pkl"))
    y_test = joblib.load(os.path.join(MODELS_DIR, "y_test.pkl"))
    x = X_test_df.tail(5)
    y = y_test.iloc[-5:] 
    predict_rate(x,y)
