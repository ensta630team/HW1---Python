import pandas as pd 
import numpy as np

def load_data(path):
    if path.endswith('.csv'):
        df = pd.read_csv(path)
    elif path.endswith('.xlsx') or path.endswith('.xls'):
        df = pd.read_excel(path, skiprows=2)
    elif path.endswith('.txt'):
        df = np.loadtxt(path)
    else:
        return None
    
    print('[INFO] Data loaded successfully!')
    return df
