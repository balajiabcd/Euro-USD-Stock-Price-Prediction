from keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from keras import layers
from .config import UNITS, DROPOUT, LEARNING_RATE, LOSS

from tensorflow.keras.losses import Huber
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam


def build_model(X_train,y_train, X_val, y_val) -> Sequential:
    model = Sequential([layers.Input((a,1)),
                    layers.LSTM(64),
                    layers.Dense(32, activation = "relu"),
                    layers.Dense(32, activation = "relu"),
                    layers.Dense(1)
                   ])
    model.compile(
                    loss=Huber(delta=1.0),
                    optimizer=Adam(learning_rate=1e-3),
                    metrics=["mae", "mape", RootMeanSquaredError(name="rmse")])

    model.fit(X_train,y_train, validation_data = (X_val, y_val), epochs = 15)

    models = [{"model1": model, "best_model": model}]
    return models
