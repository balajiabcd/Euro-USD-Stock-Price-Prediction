# tests/test_preprocess.py
import os
import numpy as np
import pandas as pd
import joblib
from src.preprocess import make_fit_scaler, transform, save_object

def test_make_fit_scaler_and_transform_roundtrip():
    df = pd.DataFrame({"a":[1,2,3,4], "b":[10,20,30,40]}, dtype=float)
    cols = ["a","b"]
    sc = make_fit_scaler(df, cols)
    out = transform(df, cols, sc)

    # shape/labels preserved
    assert list(out.columns) == cols
    assert (out.index == df.index).all()

    # scaled to [0,1]
    assert np.isclose(out["a"].min(), 0.0)
    assert np.isclose(out["a"].max(), 1.0)
    assert np.isclose(out["b"].min(), 0.0)
    assert np.isclose(out["b"].max(), 1.0)

def test_transform_uses_given_scaler_not_refitting():
    df1 = pd.DataFrame({"x":[0,1,2,3]}, dtype=float)
    df2 = pd.DataFrame({"x":[4,5,6,7]}, dtype=float)
    sc = make_fit_scaler(df1, ["x"])
    out = transform(df2, ["x"], sc)
    # values will be >1 since scaler learned min/max from df1
    assert (out["x"] > 1.0).any()

def test_save_object_writes_pkl(tmp_path):
    obj = {"k": 1}
    save_object("test_art", obj, tmp_path)
    p = tmp_path / "test_art.pkl"
    assert p.exists()
    loaded = joblib.load(p)
    assert loaded == obj
