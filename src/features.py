import pandas as pd


def build_data(df: pd.DataFrame, target_col: str):
    date, volume, high, low, ave  = [],[],[],[],[] 
    d, h, l, vol = df.date[0], 0, 0, 0
    for i in range(len(data)+1):

        if i == len(data) or d != data.date[i]:
            date.append(d)
            volume.append(vol)
            high.append(h)
            low.append(l)
            ave.append((h+l)/2)
            if i != len(data):
                d = data.date[i]
                h, l, vol = data.High[i], data.Low[i], data.Volume[i]
        else:
            vol += data.Volume[i]
            if h < data.High[i]:
                h = data.High[i]
            if l < data.Low[i]:
                l = data.Low[i]

    new_data = pd.DataFrame({ 
                            'date' : date,
                            'volume' : volume,
                            'high' : high,
                            'low': low,
                            'ave' : ave })
    return new_data
