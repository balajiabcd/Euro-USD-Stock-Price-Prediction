import pandas as pd
from .config import CSV_PATH, DATE_COL

def load_timeseries(path: str | None = None) -> pd.DataFrame:
    csv = path or str(CSV_PATH)
    df = pd.read_csv(csv)
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
        df = df.sort_values(DATE_COL).reset_index(drop=True)
    return df.dropna(how="any")
