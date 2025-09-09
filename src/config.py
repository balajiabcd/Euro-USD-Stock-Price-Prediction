from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR  = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
PLOTS_DIR  = PROJECT_ROOT / "static"

CSV_PATH  = DATA_DIR / "Entire.csv"   # put your CSV here
DATE_COL  = "Date"                    # adjust if different
TARGET_COL = "Close"                  # adjust if different

LOOKBACK = 5
PREDICT_HORIZON = 1
TEST_DAYS = 365
VAL_DAYS  = 180

EPOCHS = 40
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
