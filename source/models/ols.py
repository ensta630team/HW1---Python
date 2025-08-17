import numpy as np


class OLS:
    def __init__(self, in_dim):
        self.in_dim = in_dim 
        self.beta   = None

    # Method for adjusting parameters
    def fit(self, X, y):
        '''
        X son los rezagos que representan las variables independientes
        y es la variable dependiente
        '''
        n = X.shape[0]
        print(n)
        factor_0 = np.linalg.inv(np.matmul(X.T, X))
        factor_1 = np.matmul(X.T, y)
        self.beta = np.matmul(factor_0, factor_1)

    def predict(self, X):
        pass
       
if __name__ == '__main__':
    model = OLS(in_dim=3)
    
    beta_0 = 1
    beta_1 = 2.5
    

    model.fit(X, y)