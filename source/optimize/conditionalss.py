import numpy as np 
import copy 
from scipy.optimize import minimize

def objective_function_pq(params, y_data, p, q):
    phi = params[:p]
    theta = params[p:]

    n = len(y_data)
    errors = np.zeros(n)
    start_t = max(p, q)
    for t in range(start_t, n):
        y_past = y_data[t-p:t][::-1]
        u_past = errors[t-q:t][::-1]
        ar_part = np.dot(phi, y_past) if p > 0 else 0
        ma_part = np.dot(theta, u_past) if q > 0 else 0
        
        errors[t] = y_data[t] - ar_part - ma_part

    return np.sum(errors**2)

def conditional_square_sum(model, y_data):
        """
        Fits the ARMA(p,q) model using Conditional Sum of Squares (CSS).
        """
        # 1. Definir la función objetivo que queremos minimizar (la suma de errores al cuadrado)
        model_copy = copy.deepcopy(model)
        n = len(y_data)
        
        params_names = model_copy.params.keys()
        if 'phi' in params_names and 'theta' in params_names:
            p = len(model_copy.params['phi'])
            q = len(model_copy.params['theta'])
            initial_params = np.zeros(p + q)
            result = minimize(objective_function_pq, 
                              initial_params, 
                              args=(y_data, p, q), 
                              method='BFGS')

            # 2. variance
            sum_of_squared_residuals = result.fun
            degrees_of_freedom = n - p - q
            if degrees_of_freedom <= 0:
                error_variance = np.nan
            else:
                error_variance = sum_of_squared_residuals / degrees_of_freedom

            # 3. Calcular la desviación estándar (sigma)
            estimated_sigma = np.sqrt(error_variance)

            model_copy.ar.phi = result.x[:p]
            model_copy.ma.theta = result.x[p:]
            model_copy.sigma = estimated_sigma
            return model_copy