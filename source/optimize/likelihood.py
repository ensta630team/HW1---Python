# likelihood.py

import numpy as np
import copy
from scipy.optimize import minimize
from source.models.ols import OLS # <-- Necesitarás importar tu clase OLS

# =============================================================
# === PASO 1: Funciones para Conditional Sum of Squares (CSS) ===
# =============================================================
def objective_function_css(params, y_data, p, q):
    # Sin restricciones duras para una superficie más suave
    c = params[0]
    phi = params[1 : 1 + p]
    theta = params[1 + p :]
    
    n = len(y_data)
    errors = np.zeros(n)
    start_t = max(p, q)
    
    for t in range(start_t, n):
        y_past = y_data[t-p:t][::-1]
        u_past = errors[t-q:t][::-1]
        ar_part = np.dot(phi, y_past) if p > 0 else 0
        ma_part = np.dot(theta, u_past) if q > 0 else 0
        errors[t] = y_data[t] - c - ar_part - ma_part

    return np.sum(errors[start_t:]**2)


# =============================================================
# === PASO 2: Funciones para Maximum Likelihood (MLE) ===
# =============================================================
def objective_function_mle(params, y_data, p, q):
    # Sin restricciones duras
    c = params[0]
    phi = params[1 : 1 + p]
    theta = params[1 + p :]
    
    n = len(y_data)
    errors = np.zeros(n)
    start_t = max(p, q)
    
    for t in range(start_t, n):
        y_past = y_data[t-p:t][::-1]
        u_past = errors[t-q:t][::-1]
        ar_part = np.dot(phi, y_past) if p > 0 else 0
        ma_part = np.dot(theta, u_past) if q > 0 else 0
        errors[t] = y_data[t] - c - ar_part - ma_part

    ssr = np.sum(errors[start_t:]**2)
    if not np.isfinite(ssr):
        return np.inf

    effective_n = n - start_t
    if effective_n <= 0 or ssr <= 1e-12: # Usar un umbral pequeño para evitar log(0)
        return np.inf

    sigma2 = ssr / effective_n
    
    log_likelihood = -effective_n / 2 * (np.log(2 * np.pi) + np.log(sigma2) + 1)

    if not np.isfinite(log_likelihood):
        return np.inf

    return -log_likelihood


def maximum_likelihood_estimation(model, y_data, return_likelihood=False):
    """
    Ajusta el modelo usando CSS-MLE con mejores valores iniciales y cálculo de sigma corregido.
    """
    with np.errstate(over='ignore', invalid='ignore'):
        model_copy = copy.deepcopy(model)
        n = len(y_data)
        
        p = model_copy.p
        q = model_copy.q

        # --- PASO 1: OBTENER VALORES INICIALES CON CSS Y MEJORES GUESSES ---
        # Usar un AR(p) ajustado por OLS para los phis iniciales
        initial_phi = np.zeros(p)
        if p > 0:
            X_ar = np.array([y_data[i:i+p] for i in range(n - p)])
            y_ar = y_data[p:]
            ols = OLS()
            ols.fit(X_ar, y_ar)
            initial_phi = ols.coef_[::-1] # OLS los da en orden natural, los revertimos

        initial_theta = np.zeros(q)
        initial_c = np.mean(y_data) * (1 - np.sum(initial_phi))
        
        css_initial_params = np.concatenate(([initial_c], initial_phi, initial_theta))

        css_result = minimize(objective_function_css,
                            css_initial_params,
                            args=(y_data, p, q),
                            method='Nelder-Mead',
                            options={'maxiter': 5000}) # Aumentar iteraciones

        mle_initial_params = css_result.x
        
        # --- PASO 2: REFINAR CON MLE ---

        result = minimize(objective_function_mle,
                        mle_initial_params,
                        args=(y_data, p, q),
                        method='BFGS')

        optimal_params = result.x
        log_likelihood = -result.fun

        # --- CÁLCULO DE SIGMA CORREGIDO Y DIRECTO ---
        
        start_t = max(p, q)
        effective_n = n - start_t
        if effective_n > 0:
            # Despejar sigma^2 de la fórmula de la log-verosimilitud
            log_sigma2 = -2 * log_likelihood / effective_n - np.log(2 * np.pi) - 1
            error_variance = np.exp(log_sigma2)
            estimated_sigma = np.sqrt(error_variance)
        else:
            estimated_sigma = np.nan

        # --- ACTUALIZAR EL MODELO ---
        model_copy.c = optimal_params[0]
        model_copy.ar.c = optimal_params[0]
        model_copy.ma.c = optimal_params[0]
        model_copy.ar.phi = optimal_params[1:1+p]
        model_copy.ma.theta = optimal_params[1+p:]
        model_copy.sigma = estimated_sigma
        
        if return_likelihood:
            return model_copy, log_likelihood
        else:
            return model_copy