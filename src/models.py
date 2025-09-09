from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from .config import UNITS, DROPOUT, LEARNING_RATE, LOSS

def build_model(input_shape) -> Sequential:
    model = Sequential([
        LSTM(UNITS, input_shape=input_shape),
        Dropout(DROPOUT),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss=LOSS)
    return model
