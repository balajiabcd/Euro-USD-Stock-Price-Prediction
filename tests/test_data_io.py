# tests/test_data_io.py
import pandas as pd
import tempfile, os
from src.data_io import load_data

def test_load_data_with_datecol(tmp_path):
    p = tmp_path / "sample.csv"
    df = pd.DataFrame({
        "Date": ["2020-01-01","2020-01-02","2020-01-03"],
        "Open":[1.1,1.2,1.3],
        "Close":[1.15,1.25,1.35],
        "Volume":[100,200,150],
    })
    df.to_csv(p, index=False)

    out = load_data(str(p))

    assert "date_time_stamp" in out.columns
    assert "date" in out.columns
    assert out["date_time_stamp"].dtype == "datetime64[ns]"
    assert "Date" not in out.columns
    assert not out.isna().any().any()
