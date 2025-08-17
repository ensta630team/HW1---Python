import numpy as np
import copy
from scipy.optimize import minimize

def objective_function_mle(params, y_data, p, q):
    phi = params[:p]
    theta = params[p:]
    
    n = len(y_data)
    errors = np.zeros(n)
    
    # avoid negative starting points
    start_t = max(p, q)
    
    # Recursively calculates errors
    for t in range(start_t, n):
        y_past = y_data[t-p:t][::-1]
        u_past = errors[t-q:t][::-1]
        
        ar_part = np.dot(phi, y_past) if p > 0 else 0
        ma_part = np.dot(theta, u_past) if q > 0 else 0
        
        errors[t] = y_data[t] - ar_part - ma_part

    # Sum of Squared Residuals
    ssr = np.sum(errors[start_t:]**2)
    if not np.isfinite(ssr):
        return np.inf

    # Substract initial observations
    effective_n = n - start_t
    if effective_n <= 0 or ssr <= 1e-9: # Avoid zero div.
        return np.inf

    sigma2 = ssr / effective_n
    
    # log-likelihood 
    # L = -N/2 * log(2*pi) - N/2 * log(sigma^2) - 1/(2*sigma^2) * sum(errors^2)
    # using sigma^2 = ssr / N, la fórmula se simplifica.
    log_likelihood = -effective_n / 2 * np.log(2 * np.pi) - effective_n / 2 * np.log(sigma2) - ssr / (2 * sigma2)

    if not np.isfinite(log_likelihood):
        return np.inf

    return -log_likelihood


def maximum_likelihood_estimation(model, y_data, return_likelihood=False):
    """
    Fits the model using Maximum Likelihood Estimation (MLE).
    """
    model_copy = copy.deepcopy(model)
    n = len(y_data)
    
    params_names = model_copy.params.keys()
    if 'phi' in params_names and 'theta' in params_names:
        p = model_copy.p
        q = model_copy.q
        
        # Parámetros iniciales para el optimizador (un vector de ceros es un comienzo simple)
        initial_params = np.zeros(p + q)
        
        # 2. Minimizar el negativo de la log-verosimilitud
        result = minimize(objective_function_mle,
                          initial_params,
                          args=(y_data, p, q),
                          method='BFGS') # BFGS es un método de optimización robusto

        # Coeficientes óptimos encontrados
        optimal_params = result.x
        phi_estimated = optimal_params[:p]
        theta_estimated = optimal_params[p:]
        log_likelihood = -result.fun

        # 3. Calcular la desviación estándar (sigma) final con los parámetros óptimos
        # Para ello, necesitamos recalcular los residuos finales.
        errors = np.zeros(n)
        start_t = max(p, q)
        for t in range(start_t, n):
            y_past = y_data[t-p:t][::-1]
            u_past = errors[t-q:t][::-1]
            ar_part = np.dot(phi_estimated, y_past) if p > 0 else 0
            ma_part = np.dot(theta_estimated, u_past) if q > 0 else 0
            errors[t] = y_data[t] - ar_part - ma_part
            
        sum_of_squared_residuals = np.sum(errors[start_t:]**2)
        degrees_of_freedom = n - start_t
        
        if degrees_of_freedom <= 0:
            error_variance = np.nan
        else:
            # La estimación MLE de la varianza
            error_variance = sum_of_squared_residuals / degrees_of_freedom

        estimated_sigma = np.sqrt(error_variance)

        # 4. Actualizar el modelo con los parámetros estimados
        model_copy.ar.phi = phi_estimated
        model_copy.ma.theta = theta_estimated
        model_copy.sigma = estimated_sigma
        if return_likelihood:
            return model_copy, log_likelihood
        else:
            return model_copy
    