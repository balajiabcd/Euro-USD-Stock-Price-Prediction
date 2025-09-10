# src/plot_chart.py
import os, json, joblib
import numpy as np
import matplotlib.pyplot as plt
from .config import MODELS_DIR, PROJECT_ROOT

# save plots under static/plots/
PLOT_DIR = os.path.join(PROJECT_ROOT, "static", "plots")


def _load_artifact(name):
    return joblib.load(os.path.join(MODELS_DIR, f"{name}.pkl"))


def _ensure_dirs():
    os.makedirs(PLOT_DIR, exist_ok=True)


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


def make_all_plots():
    _ensure_dirs()
    y_true = _load_artifact("y_test")
    y_pred = _load_artifact("y_pred_test")
    y_true = y_true.values if hasattr(y_true, "values") else np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Load metrics to show in console
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        print("Metrics:", metrics)

    plot_actual_vs_pred(y_true, y_pred)
    plot_residuals(y_true, y_pred)
    plot_parity(y_true, y_pred)
    print(f"Plots saved in: {PLOT_DIR}")


if __name__ == "__main__":
    make_all_plots()
