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

def create_dataset(path, problem=3):
    df = load_data(path)

    if problem == 3:
        pi_t = np.log(df['IPC'] / df['IPC'].shift(1))
        pi_t.dropna(inplace=True)
        pi_t = pi_t.to_numpy()
        
        y_t = np.log(df['IMACEC'] / df['IMACEC'].shift(1))
        y_t.dropna(inplace=True)
        y_t = y_t.to_numpy()

        i_t = df['Tasa de política'] / 100
        i_t = i_t.to_numpy()
        
        t = df['Periodo'].dt.date
        t = t.to_numpy()
        return {
            't': t,
            'pi_t': pi_t,
            'y_t': y_t,
            'i_t': i_t
        }
    if problem == 4: 
        return df