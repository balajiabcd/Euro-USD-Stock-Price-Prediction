# tests/test_models.py
import numpy as np
import pandas as pd
import pytest

import src.models as models


def make_fake_data(n_samples=10, n_features=4):
    X = pd.DataFrame(
        np.random.rand(n_samples, n_features),
        columns=[f"f{i}" for i in range(n_features)]
    )
    y = pd.Series(np.random.rand(n_samples))
    return X, y


def test_build_models_returns_dict(monkeypatch):
    X_train, y_train = make_fake_data()
    X_val, y_val = make_fake_data()

    # Monkeypatch model.fit to avoid long training
    def fake_fit(self, X, y, validation_data=None, epochs=1):
        return None
    monkeypatch.setattr(models.Sequential, "fit", fake_fit)

    result = models.build_models(X_train, y_train, X_val, y_val, epochs=1)

    assert isinstance(result, dict)
    assert "model1" in result
    assert "best_model" in result
    assert result["model1"] is result["best_model"]

    model = result["model1"]
    # Model should have keras-like methods
    assert hasattr(model, "compile")
    assert hasattr(model, "fit")
