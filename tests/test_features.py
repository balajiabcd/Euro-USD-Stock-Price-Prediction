# tests/test_features.py
import pandas as pd
import numpy as np
from src.features import refine_data, build_data

def test_refine_data_aggregations():
    df = pd.DataFrame({
        "date": pd.to_datetime([
            "2020-01-01 09:00", "2020-01-01 12:00", "2020-01-01 17:00",
            "2020-01-02 09:00", "2020-01-02 17:00"
        ]).date,
        "Open":  [1.10, 1.12, 1.11, 1.20, 1.22],
        "Close": [1.11, 1.13, 1.14, 1.21, 1.25],
        "High":  [1.12, 1.15, 1.14, 1.23, 1.26],
        "Low":   [1.09, 1.10, 1.10, 1.19, 1.20],
        "Volume":[100, 150, 200, 120, 130],
    })
    out = refine_data(df)

    # One row per day
    assert list(out["date"]) == sorted(out["date"].tolist())
    assert len(out) == 2

    # Day 1 aggregations
    d1 = out.iloc[0]
    assert np.isclose(d1["open"], 1.10)      # first Open of the day
    assert np.isclose(d1["close"], 1.14)     # last Close of the day
    assert np.isclose(d1["high"], 1.15)      # max High
    assert np.isclose(d1["low"],  1.09)      # min Low
    assert d1["volume"] == 450               # sum Volume
    assert np.isclose(d1["ave"], (1.15+1.09)/2)

def test_build_data_sliding_window():
    # Simple target series to verify windowing
    df = pd.DataFrame({"ave": [1, 2, 3, 4, 5]})
    n = 3
    out = build_data(df, "ave", n)

    # Expect rows = len(values) - n
    assert len(out) == 5 - n  # 2 rows
    # Columns day_0..day_{n-1} + next_day
    assert list(out.columns) == [f"day_{i}" for i in range(n)] + ["next_day"]

    # Check actual window content
    expected = pd.DataFrame({
        "day_0": [1, 2],
        "day_1": [2, 3],
        "day_2": [3, 4],
        "next_day": [4, 5],
    })
    pd.testing.assert_frame_equal(out.reset_index(drop=True), expected)
