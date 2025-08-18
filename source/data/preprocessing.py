import numpy as np 
                                     
def to_X_y(y, p):
    nobs = len(y)
    X = np.ones(nobs-p)
    for i in range(p):
        nuevorezago = y[p-1-i:nobs-1-i]
        X = np.column_stack([X, nuevorezago])
    Y = y[p:]
    return X, Y

def split_serie(serie, limit='2023-12-01'):

    pass