# tests/test_predict.py
import numpy as np
import pandas as pd
import pytest

import src.predict as predict


def test_predict_rate_prints(monkeypatch, capsys):
    # Fake test data
    X = np.array([[1.0], [2.0], [3.0]])
    y = pd.Series([1.1, 2.1, 3.1])

    # Patch global y_test inside module
    predict.y_test = y

    # Dummy model with .predict
    class DummyModel:
        def predict(self, X_in):
            return np.array([[1.0], [2.0], [3.0]])

    monkeypatch.setattr(predict, "load_model", lambda path: DummyModel())

    # Call function
    predict.predict_rate(X, y)

    out = capsys.readouterr().out
    # Expect lines with "original" and "prediction"
    assert "original is" in out
    assert "prediciton is" in out  # note: file has typo "prediciton"
    # Check count matches input length
    assert len([l for l in out.splitlines() if "original" in l]) == len(X)
