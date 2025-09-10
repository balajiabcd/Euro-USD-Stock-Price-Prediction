# src/plot_chart.py
import os, json, joblib
import numpy as np
import matplotlib.pyplot as plt
from .config import MODELS_DIR, PROJECT_ROOT

# save plots under static/plots/
PLOT_DIR = os.path.join(PROJECT_ROOT, "static", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


def load_artifact(name):
    return joblib.load(os.path.join(MODELS_DIR, f"{name}.pkl"))


def plot_actual_vs_pred(y_true, y_pred):
    plt.figure(figsize=(9,4.5))
    plt.plot(y_true, label="Actual")
    plt.plot(y_pred, label="Predicted")
    plt.title("Actual vs Predicted (Test Set)")
    plt.xlabel("Sample")
    plt.ylabel("Target")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "01_actual_vs_pred.png"), dpi=150)
    plt.close()


def plot_residuals(y_true, y_pred):
    res = y_true - y_pred
    plt.figure(figsize=(9,4.5))
    plt.plot(res, color="tab:red")
    plt.axhline(0, color="black", linewidth=1)
    plt.title("Residuals Over Time")
    plt.xlabel("Sample")
    plt.ylabel("Residual")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "02_residuals.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(6,4.5))
    plt.hist(res, bins=30, color="tab:blue", alpha=0.7)
    plt.title("Residual Distribution")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "03_residual_hist.png"), dpi=150)
    plt.close()


def plot_parity(y_true, y_pred):
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, s=20, alpha=0.6)
    plt.plot(lims, lims, 'k--', lw=2)
    plt.title("Parity Plot (Pred vs Actual)")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "04_parity.png"), dpi=150)
    plt.close()


def plot_naive_baseline(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    # Naive: previous actual as prediction (shift by 1)
    naive = y_true[:-1]
    actual = y_true[1:]
    model  = y_pred[1:]  # align with 'actual'

    # Metrics (printed to console)
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    rmse_model = np.sqrt(mean_squared_error(actual, model))
    rmse_naive = np.sqrt(mean_squared_error(actual, naive))
    mae_model  = mean_absolute_error(actual, model)
    mae_naive  = mean_absolute_error(actual, naive)
    impr = 100 * (rmse_naive - rmse_model) / rmse_naive if rmse_naive > 0 else 0.0
    print({"baseline_rmse": rmse_naive, "model_rmse": rmse_model, "baseline_mae": mae_naive, "model_mae": mae_model, "rmse_improvement_%": impr})

    # Plot
    plt.figure(figsize=(9,4.5))
    plt.plot(actual, label="Actual")
    plt.plot(model,  label="Model")
    plt.plot(naive,  label="Naïve (y[t-1])", alpha=0.7)
    plt.title("Model vs Naïve Baseline")
    plt.xlabel("Aligned sample")
    plt.ylabel("Target")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "05_model_vs_naive.png"), dpi=150)
    plt.close()


def plot_rolling_mae(y_true, y_pred, window=20):
    """Rolling MAE to show stability over time."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    err = np.abs(y_true - y_pred)

    if len(err) < window:
        window = max(3, len(err)//5)

    roll = np.convolve(err, np.ones(window)/window, mode="valid")
    plt.figure(figsize=(9,4.5))
    plt.plot(roll)
    plt.title(f"Rolling MAE (window={window})")
    plt.xlabel("Sample")
    plt.ylabel("MAE")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "06_rolling_mae.png"), dpi=150)
    plt.close()


def plot_residual_autocorr(y_true, y_pred, max_lag=30):
    """Simple autocorrelation of residuals up to max_lag."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    r = y_true - y_pred
    r = r - r.mean()
    denom = np.dot(r, r) if np.dot(r, r) != 0 else 1.0

    lags = np.arange(1, max_lag+1)
    acf = []
    for k in lags:
        num = np.dot(r[:-k], r[k:])
        acf.append(num / denom)

    plt.figure(figsize=(9,4.5))
    plt.bar(lags, acf, width=0.8)
    plt.axhline(0, color="black", linewidth=1)
    plt.title("Residual Autocorrelation")
    plt.xlabel("Lag")
    plt.ylabel("ACF")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "07_residual_acf.png"), dpi=150)
    plt.close()


def plot_error_vs_level(y_true, y_pred):
    """Scatter of residuals vs actual level to spot heteroscedasticity."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    res = y_true - y_pred

    plt.figure(figsize=(9,4.5))
    plt.scatter(y_true, res, s=18, alpha=0.6)
    plt.axhline(0, color="black", linewidth=1)
    plt.title("Residual vs Actual Level")
    plt.xlabel("Actual")
    plt.ylabel("Residual (Actual - Pred)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "08_residual_vs_level.png"), dpi=150)
    plt.close()


def plot_abs_error_by_level_bins(y_true, y_pred, n_bins=5):
    """Boxplots of |error| grouped by actual-level quantile bins."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    ae = np.abs(y_true - y_pred)

    # quantile bins on actual
    qs = np.quantile(y_true, np.linspace(0, 1, n_bins+1))
    bins = np.digitize(y_true, qs[1:-1], right=True)  # groups 0..n_bins-1

    data = [ae[bins == i] for i in range(n_bins)]
    labels = [f"[{qs[i]:.3f},{qs[i+1]:.3f}]" for i in range(n_bins)]

    plt.figure(figsize=(10,4.5))
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.title("Absolute Error by Actual-Level Quantile")
    plt.xlabel("Actual level bin")
    plt.ylabel("|Error|")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "09_abs_error_by_level_bins.png"), dpi=150)
    plt.close()


def make_all_plots():
    y_true = load_artifact("y_test")
    y_pred = load_artifact("y_pred_test")
    y_true = y_true.values if hasattr(y_true, "values") else np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    # Print metrics if available
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        print("Metrics:", metrics)

    plot_actual_vs_pred(y_true, y_pred)
    plot_residuals(y_true, y_pred)
    plot_parity(y_true, y_pred)
    plot_naive_baseline(y_true, y_pred)
    plot_rolling_mae(y_true, y_pred, window=20)
    plot_residual_autocorr(y_true, y_pred, max_lag=30)
    plot_error_vs_level(y_true, y_pred)
    plot_abs_error_by_level_bins(y_true, y_pred, n_bins=5)

    print(f"Plots saved in: {PLOT_DIR}")



if __name__ == "__main__":
    make_all_plots()
