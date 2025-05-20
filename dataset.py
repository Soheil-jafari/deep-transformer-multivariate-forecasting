import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch

def load_data(path, seq_len=24, pred_len=12):
    df = pd.read_csv(path)
    scaler = MinMaxScaler()
    data = scaler.fit_transform(df.values)

    X, Y = [], []
    for i in range(len(data) - seq_len - pred_len):
        X.append(data[i:i+seq_len])
        Y.append(data[i+seq_len:i+seq_len+pred_len])

    X = torch.tensor(np.array(X), dtype=torch.float32)
    Y = torch.tensor(np.array(Y), dtype=torch.float32)
    return X, Y, scaler
