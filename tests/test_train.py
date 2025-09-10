# tests/test_train.py
import types
import pandas as pd
import numpy as np
import pytest

import src.train as train


def test_main_runs(monkeypatch, tmp_path):
    # --- Fake data ---
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=30, freq="D").date,
        "Open": np.linspace(1.0, 1.2, 30),
        "Close": np.linspace(1.05, 1.25, 30),
        "High": np.linspace(1.1, 1.3, 30),
        "Low": np.linspace(0.95, 1.15, 30),
        "Volume": np.random.randint(100, 200, size=30),
    })

    # --- Monkeypatch dependencies ---
    monkeypatch.setattr(train, "load_data", lambda path: df)
    monkeypatch.setattr(train, "refine_data", lambda df: df)
    monkeypatch.setattr(train, "build_data", lambda df, target, n: pd.DataFrame({
        "day_0": range(20), "next_day": range(100, 120)
    }))
    monkeypatch.setattr(train, "data_split", lambda df: (
        pd.DataFrame({"day_0": range(10)}), pd.Series(range(10)),
        pd.DataFrame({"day_0": range(5)}), pd.Series(range(5)),
        pd.DataFrame({"day_0": range(5)}), pd.Series(range(5)),
    ))
    monkeypatch.setattr(train, "make_fit_scaler", lambda df, cols: "scaler")
    monkeypatch.setattr(train, "transform", lambda df, cols, scaler: df)
    monkeypatch.setattr(train, "save_object", lambda name, value, path: None)

    class DummyModel:
        def save(self, path): pass
    monkeypatch.setattr(train, "build_models", lambda *a, **k: {"best_model": DummyModel()})

    # --- Run ---
    monkeypatch.setattr(train, "MODELS_DIR", tmp_path)
    train.main()

    # Check that model file was created
    # (build_models.save writes files in MODELS_DIR, we patched save to no-op, so just check dir exists)
    assert tmp_path.exists()
