import pandas as pd

def refine_data(df: pd.DataFrame):
    daily_data = df.groupby("date").agg(
        open   = ("Open", "first"),   # first price of the day
        close  = ("Close", "last"),   # last price of the day
        high   = ("High", "max"),     # max price of the day
        low    = ("Low", "min"),      # min price of the day
        volume = ("Volume", "sum")    # sum of volume
    ).reset_index()                   # keep 'date' as a column
    daily_data["ave"] = (daily["high"] + daily["low"]) / 2    # Add average of high & low
    return daily_data


def build_data(df, TARGET_COL, n):
    values = df[target_col].values
    data = [values[i-n:i+1] for i in range(n, len(values))]
    cols = [f"day_{i}" for i in range(n)] + ["next_day"]
    df = pd.DataFrame(data, columns=cols)
    return df