
import numpy as np
import pandas as pd
import pytest

@pytest.fixture
def sample_series():
    return pd.Series([1.0, 2.0, 3.0, 4.0])

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Open": [1.10, 1.11, 1.12, 1.13, 1.14],
        "High": [1.11, 1.12, 1.13, 1.14, 1.15],
        "Low":  [1.09, 1.10, 1.11, 1.12, 1.13],
        "Close":[1.105,1.115,1.125,1.135,1.145],
        "Volume":[100,120,130,110,90]
    })
