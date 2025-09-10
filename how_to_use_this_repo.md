# How to Use This Repo (Euro-USD-Stock-Price-Prediction)

> Euro‑USD Stock Price Prediction (LSTM) — quick start guide.

## 1) Setup

```bash
# Clone and enter
cd Euro-USD-Stock-Price-Prediction

# (Recommended) Create & activate env (name used in chats)
python -m venv env_euro_usd_stock
# macOS/Linux
source env_euro_usd_stock/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

**Python:** 3.10–3.11 is recommended.  
**Run from the repo root** for all commands below (we use `-m src.<module>`).

## 2) Data

- Put your CSV at: `data/Entire.csv` (default expected by `src/config.py`).
- You can find the data used in this project [Here](https://www.kaggle.com/datasets/ahmed121ashraf131/euro-and-usd-stocks-data-2007-2021).
- The loader will parse a `Date` column and create `date_time_stamp` and `date`.

```
data/
└── Entire.csv
```

## 3) Train

```bash
# From repo root
python -m src.train
```
This will:
- load & refine data (`src/data_io.py`, `src/features.py`)
- build lookback features (see `LOOKBACK_days` in `src/config.py`)
- split, scale, and train
- save artifacts in `models/`:
  - `best_model.keras`, `X_*_df.pkl`, `y_* .pkl`, `scaler.pkl`

## 4) Evaluate

```bash
python -m src.evaluate
```
Produces metrics (`models/metrics.json`) and `models/y_pred_test.pkl`.

## 5) Predict (quick demo)

```bash
python -m src.predict
```
By default it prints last 5 test samples: actual vs prediction.

## 6) Plots

```bash
python -m src.plot_chart
```
Generates figures under `static/plots/`:
`01_actual_vs_pred.png … 09_abs_error_by_level_bins.png`

## 7) Configuration

Most knobs live in **`src/config.py`** (paths, lookback, epochs, etc.).  
Key ones:
- `CSV_PATH` → `data/Entire.csv`
- `LOOKBACK_days` (window length)
- `EPOCHS`, `BATCH_SIZE`
- output dirs: `MODELS_DIR`, `PLOTS_DIR`

## 8) Tests (optional)

```bash
# Make sure you are at repo root
pytest -q
```
If needed, set Python path:
```bash
# Alternative to -m: add Python path for local runs
# Windows (PowerShell)
$env:PYTHONPATH = "$PWD"
# macOS/Linux
export PYTHONPATH="$PWD"
```

## 9) Common issues

- **Module import errors:** Always run with `python -m src.<name>` from **repo root**.
- **Missing data:** Ensure `data/Entire.csv` exists; adjust `CSV_PATH` if different.
- **TensorFlow not found:** Re‑install per `requirements.txt` inside the same env.
- **GPU vs CPU:** This setup works on CPU; GPU is optional.

---

**Tip:** Keep results tidy — `models/` for artifacts, `static/plots/` for charts.
