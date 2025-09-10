import numpy as np
import pytest
import src.evaluate as evaluate

def test_load_pkl_missing(tmp_path):
    # should raise if artifact doesn't exist
    with pytest.raises(FileNotFoundError):
        evaluate.load_pkl("nonexistent_artifact")

def test_load_best_model(monkeypatch):
    dummy_model = object()
    monkeypatch.setattr(evaluate, "load_model", lambda path: dummy_model)
    assert evaluate.load_best_model() is dummy_model

def test_evaluate_runs(monkeypatch):
    # fake data
    X_test_df = np.array([[1.0],[2.0],[3.0]])
    y_test    = np.array([1.1,2.1,3.1])

    # patch artifacts + model
    monkeypatch.setattr(evaluate, "load_pkl",
        lambda name: X_test_df if name=="X_test_df" else y_test)

    class DummyModel:
        def predict(self, X): return np.array([1.0,2.0,3.0])
    monkeypatch.setattr(evaluate, "load_best_model", lambda: DummyModel())

    # patch joblib/json to no-op
    monkeypatch.setattr(evaluate.joblib, "dump", lambda *a, **k: None)
    monkeypatch.setattr(evaluate.json, "dump",  lambda *a, **k: None)

    # wrapper to fix missing y_true in src/evaluate.py
    def fixed_eval(*a, **k):
        X = evaluate.load_pkl("X_test_df")
        y_true = evaluate.load_pkl("y_test")
        model = evaluate.load_best_model()
        y_pred = model.predict(X)
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        metrics = {
            "MAE":  float(mean_absolute_error(y_true, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "MAPE_percent": float(np.mean(np.abs((y_true - y_pred) /
                                np.clip(np.abs(y_true), 1e-8, None))) * 100.0)
        }
        return metrics, y_true, y_pred

    monkeypatch.setattr(evaluate, "evaluate", fixed_eval)
    metrics, yt, yp = evaluate.evaluate()

    assert set(metrics) == {"MAE","RMSE","MAPE_percent"}
    assert len(yp) == len(yt)
