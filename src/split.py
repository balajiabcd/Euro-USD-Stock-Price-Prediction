import pandas as pd
from .config import DATE_COL, TEST_DAYS, VAL_DAYS

def temporal_split(df: pd.DataFrame):
    if DATE_COL in df.columns:
        max_date = df[DATE_COL].max()
        test_start = max_date - pd.Timedelta(days=TEST_DAYS)
        val_start  = test_start - pd.Timedelta(days=VAL_DAYS)
        train = df[df[DATE_COL] < val_start]
        val   = df[(df[DATE_COL] >= val_start) & (df[DATE_COL] < test_start)]
        test  = df[df[DATE_COL] >= test_start]
    else:
        n = len(df)
        n_test = max(1, int(0.15*n))
        n_val  = max(1, int(0.15*n))
        train = df.iloc[: n - n_val - n_test]
        val   = df.iloc[n - n_val - n_test : n - n_test]
        test  = df.iloc[n - n_test : ]
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
