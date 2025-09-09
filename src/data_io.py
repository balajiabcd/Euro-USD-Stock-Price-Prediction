import pandas as pd
from .config import CSV_PATH, DATE_COL

def load_data(path = str(CSV_PATH)):
    df = pd.read_csv(csv)
    if DATE_COL in df.columns:
        df["date_time_stamp"] = pd.to_datetime(df[DATE_COL], errors="coerce")
        df["date"] = df["date_time_stamp"].dt.date
        df = df.sort_values("date_time_stamp").reset_index(drop=True)
        df = df.drop(columns=[DATE_COL])
    return df.dropna(how="any")
