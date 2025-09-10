import pandas as pd

def data_split(df: pd.DataFrame, train_size=0.7, val_size=0.15):
    n_train = int(len(df) * train_size)
    n_val = int(len(df) * val_size)
    
    train = df.iloc[:n_train]
    val   = df.iloc[n_train:n_train+n_val]
    test  = df.iloc[n_train+n_val:]

    return (
    train.drop(columns=["next_day"]), train["next_day"], 
    val.drop(columns=["next_day"]),   val["next_day"], 
    test.drop(columns=["next_day"]),  test["next_day"]  )
