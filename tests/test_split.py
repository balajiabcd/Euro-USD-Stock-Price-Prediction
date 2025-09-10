# tests/test_split.py
import pandas as pd
from src.split import data_split

def make_df(n=10):
    # simple dummy dataset
    return pd.DataFrame({
        "day_0": range(n),
        "day_1": range(n, 2*n),
        "next_day": range(100, 100+n)
    })

def test_data_split_shapes_and_order():
    df = make_df(20)
    X_train, y_train, X_val, y_val, X_test, y_test = data_split(df, train_size=0.5, val_size=0.25)

    # Train = 50%, Val = 25%, Test = rest
    assert len(X_train) == 10
    assert len(X_val) == 5
    assert len(X_test) == 5

    # Columns should not include "next_day"
    assert "next_day" not in X_train.columns
    assert "next_day" not in X_val.columns
    assert "next_day" not in X_test.columns

    # Targets should equal original "next_day" values
    assert y_train.iloc[0] == 100
    assert y_val.iloc[0] == 110
    assert y_test.iloc[0] == 115

def test_split_respects_order():
    df = make_df(12)
    _, y_train, _, y_val, _, y_test = data_split(df, train_size=0.5, val_size=0.25)
    # Ensure no shuffling: strictly sequential
    assert list(y_train) + list(y_val) + list(y_test) == list(df["next_day"])
