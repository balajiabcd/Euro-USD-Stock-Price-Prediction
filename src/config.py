from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR  = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
PLOTS_DIR  = PROJECT_ROOT / "plots"

CSV_PATH  = DATA_DIR / "Entire.csv"
DATE_COL  = "Date"
TARGET_COL = "ave"

LOOKBACK_days = 45

EPOCHS = 20
BATCH_SIZE = 32
UNITS   = 64
DROPOUT = 0.2
LEARNING_RATE = 1e-3
RANDOM_SEED = 42

LOSS = "mse"
OPTIMIZER = "adam"
BIDIRECTIONAL = False
N_LSTM_LAYERS = 1
DENSE_HEAD = False
