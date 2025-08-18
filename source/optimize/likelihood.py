# likelihood.py

import numpy as np
import copy
from scipy.optimize import minimize

# =============================================================
# === PASO 1: Funciones para Conditional Sum of Squares (CSS) ===
# =============================================================
def objective_function_css(params, y_data, p, q):
    """
    Calcula la suma de los cuadrados de los errores para la estimación CSS.
    Esta función es más simple y estable que la de MLE.
    """
    c = params[0]
    phi = params[1 : 1 + p]
    theta = params[1 + p :]

    # Penalizar parámetros que resultan en un modelo no estacionario o no invertible
    # Comprobación de estacionariedad (raíces del polinomio AR fuera del círculo unitario)
    if p > 0:
        if np.any(np.abs(np.roots(np.r_[1, -phi])) <= 1):
            return 1e9 # Devolver un valor muy grande

    # Comprobación de invertibilidad (raíces del polinomio MA fuera del círculo unitario)
    if q > 0:
        if np.any(np.abs(np.roots(np.r_[1, theta])) <= 1):
            return 1e9 # Devolver un valor muy grande
    
    n = len(y_data)
    errors = np.zeros(n)
    
    start_t = max(p, q)
    
    for t in range(start_t, n):
        y_past = y_data[t-p:t][::-1]
        u_past = errors[t-q:t][::-1]
        
        ar_part = np.dot(phi, y_past) if p > 0 else 0
        ma_part = np.dot(theta, u_past) if q > 0 else 0
        
        errors[t] = y_data[t] - c - ar_part - ma_part

    # Simplemente devolvemos la suma de los errores al cuadrado
    return np.sum(errors[start_t:]**2)


def objective_function_mle(params, y_data, p, q):
    c = params[0]
    phi = params[1 : 1 + p]
    theta = params[1 + p :]

    if p > 0:
        if np.any(np.abs(np.roots(np.r_[1, -phi])) <= 1):
            return np.inf # Devolver infinito para que el optimizador evite esta zona

    if q > 0:
        if np.any(np.abs(np.roots(np.r_[1, theta])) <= 1):
            return np.inf # Devolver infinito

    n = len(y_data)
    errors = np.zeros(n)
    start_t = max(p, q)
    
    for t in range(start_t, n):
        y_past = y_data[t-p:t][::-1]
        u_past = errors[t-q:t][::-1]
        ar_term = np.dot(phi, y_past) if p > 0 else 0
        ma_term = np.dot(theta, u_past) if q > 0 else 0
        errors[t] = y_data[t] - c - ar_term - ma_term

    ssr = np.sum(errors[start_t:]**2)
    
    if not np.isfinite(ssr):
        return np.inf

    effective_n = n - start_t
    if effective_n <= 0 or ssr <= 1e-9:
        return np.inf

    sigma2 = ssr / effective_n
    
    log_likelihood = -effective_n / 2 * np.log(2 * np.pi) - effective_n / 2 * np.log(sigma2) - ssr / (2 * sigma2)

    if not np.isfinite(log_likelihood):
        return np.inf

    return -log_likelihood


def maximum_likelihood_estimation(model, y_data, return_likelihood=False):
    """
    Ajusta el modelo usando la estrategia CSS-MLE en dos pasos.
    """
    model_copy = copy.deepcopy(model)
    n = len(y_data)
    
    params_names = model_copy.params.keys()
    if 'phi' in params_names and 'theta' in params_names:
        p = model_copy.p
        q = model_copy.q

        # --- PASO 1: OBTENER VALORES INICIALES CON CSS ---
        # Usar los parámetros del modelo (de Hannan-Rissanen) como valores iniciales
        css_initial_params = np.concatenate([
            [model_copy.c],
            model_copy.phi,
            model_copy.theta
        ])
        css_result = minimize(objective_function_css,
                              css_initial_params,
                              args=(y_data, p, q),
                              method='Nelder-Mead')
        
        # Los resultados de CSS son ahora nuestros valores iniciales para MLE
        mle_initial_params = css_result.x
        
        # --- PASO 2: REFINAR CON MLE ---
        result = minimize(objective_function_mle,
                          mle_initial_params, # <-- Usando los mejores valores iniciales
                          args=(y_data, p, q),
                          method='BFGS')

        # El resto del código para extraer y actualizar el modelo es el mismo
        optimal_params = result.x
        c_estimated = optimal_params[0]
        phi_estimated = optimal_params[1:1+p]
        theta_estimated = optimal_params[1+p:]
        log_likelihood = -result.fun

        errors = np.zeros(n)
        start_t = max(p, q)
        for t in range(start_t, n):
            y_past = y_data[t-p:t][::-1]
            u_past = errors[t-q:t][::-1]
            ar_part = np.dot(phi_estimated, y_past) if p > 0 else 0
            ma_part = np.dot(theta_estimated, u_past) if q > 0 else 0
            errors[t] = y_data[t] - c_estimated- ar_part - ma_part
            
        sum_of_squared_residuals = np.sum(errors[start_t:]**2)
        # El número de parámetros estimados es ahora p + q + c = k_params
        k_params = p + q + 1
        degrees_of_freedom = n - start_t
        
        if degrees_of_freedom <= 0:
            error_variance = np.nan
        else:
            # La estimación MLE de la varianza es SSR/n
            error_variance = sum_of_squared_residuals / degrees_of_freedom

        estimated_sigma = np.sqrt(error_variance)

        model_copy.c = c_estimated
        model_copy.ar.c = c_estimated
        model_copy.ma.c = c_estimated
        model_copy.ar.phi = phi_estimated
        model_copy.ma.theta = theta_estimated
        model_copy.sigma = estimated_sigma
        
        if return_likelihood:
            return model_copy, log_likelihood
        else:
            return model_copy