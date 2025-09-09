import pandas as pd

def add_technical_indicators(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    out = df.copy()
    if target_col not in out.columns:
        return out
    out["SMA_5"]  = out[price_col].rolling(5, min_periods=1).mean()
    out["SMA_10"] = out[price_col].rolling(10, min_periods=1).mean()
    out["EMA_5"]  = out[price_col].ewm(span=5, adjust=False).mean()
    out["EMA_10"] = out[price_col].ewm(span=10, adjust=False).mean()
    out["RET_1"]  = out[price_col].pct_change(1)
    out["DIFF_1"] = out[price_col].diff(1)
    out["LAG_1"]  = out[price_col].shift(1)
    out["LAG_2"]  = out[price_col].shift(2)
    out = out.dropna().reset_index(drop=True)
    return out
