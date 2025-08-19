import numpy as np 
import pandas as pd
from datetime import datetime


def to_X_y(y, p):
    nobs = len(y)
    X = np.ones(nobs-p)
    for i in range(p):
        nuevorezago = y[p-1-i:nobs-1-i]
        X = np.column_stack([X, nuevorezago])
    Y = y[p:]
    return X, Y

def split_dataset(dataset: dict, cutoff_date_str: str) -> dict:
    dataset_cp = dataset.copy()
    dataset_cp['t'] = dataset_cp['t'][1:]
    dataset_cp['i_t'] = dataset_cp['i_t'][1:]
    
    df = pd.DataFrame(dataset_cp)
    df['t'] = pd.to_datetime(df['t'])

    df_0 = df[df['t'] <= cutoff_date_str]
    df_1 = df[df['t'] > cutoff_date_str] 

    return df_0.to_dict(orient='list'), df_1.to_dict(orient='list')