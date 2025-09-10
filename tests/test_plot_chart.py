# tests/test_plot_chart.py
import os
os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg")

import numpy as np
from pathlib import Path
import src.plot_chart as pc

def _y():
    y_true = np.linspace(1.0, 2.0, 50)
    rng = np.random.default_rng(0)
    y_pred = y_true + rng.normal(0, 0.01, size=len(y_true))
    return y_true, y_pred

def test_each_plot_writes_file(tmp_path, monkeypatch):
    # send outputs to tmp
    monkeypatch.setattr(pc, "PLOT_DIR", str(tmp_path))
    y_true, y_pred = _y()

    pc.plot_actual_vs_pred(y_true, y_pred)
    pc.plot_residuals(y_true, y_pred)
    pc.plot_parity(y_true, y_pred)
    pc.plot_naive_baseline(y_true, y_pred)
    pc.plot_rolling_mae(y_true, y_pred, window=10)
    pc.plot_residual_autocorr(y_true, y_pred, max_lag=10)
    pc.plot_error_vs_level(y_true, y_pred)
    pc.plot_abs_error_by_level_bins(y_true, y_pred, n_bins=4)

    expected = [
        "01_actual_vs_pred.png",
        "02_residuals.png",
        "03_residual_hist.png",
        "04_parity.png",
        "05_model_vs_naive.png",
        "06_rolling_mae.png",
        "07_residual_acf.png",
        "08_residual_vs_level.png",
        "09_abs_error_by_level_bins.png",
    ]
    for name in expected:
        assert (tmp_path / name).exists(), f"missing {name}"

def test_make_all_plots_with_mocked_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "PLOT_DIR", str(tmp_path))
    y_true, y_pred = _y()
    monkeypatch.setattr(pc, "load_artifact", lambda name: y_true if name=="y_test" else y_pred)

    pc.make_all_plots()

    # spot-check a few outputs
    for name in ["01_actual_vs_pred.png", "05_model_vs_naive.png", "09_abs_error_by_level_bins.png"]:
        assert (tmp_path / name).exists()
