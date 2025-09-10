from keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from keras import layers
from .config import UNITS, DROPOUT, LEARNING_RATE, LOSS

from tensorflow.keras.losses import Huber
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy

def build_models(X_train,y_train, X_val, y_val, epochs):
    if X_train.ndim == 2:  # (N, T) → (N, T, 1)
        X_train = X_train[..., None]
        X_val   = X_val[..., None]

    T = X_train.shape[1]
    F = X_train.shape[2]

    model1 = Sequential([   layers.Input((T, F)),
                            layers.LSTM(64),
                            layers.Dense(32, activation = "relu"),
                            layers.Dense(32, activation = "relu"),
                            layers.Dense(1) ])
    model1.compile(
                    loss=Huber(delta=1.0),
                    optimizer=Adam(learning_rate=1e-3),
                    metrics=["mae", "mape", RootMeanSquaredError(name="rmse")])
    
    cbs = [ EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.5, verbose=0) ]

    model1.fit(X_train,y_train, validation_data = (X_val, y_val), epochs = epochs, callbacks=cbs)
    

    models = {"model1": model1, "best_model": model1}
    return models
