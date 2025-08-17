import numpy as np
from scipy.optimize import minimize
from scipy.linalg import solve_discrete_lyapunov
from source.optimize.conditionalss import conditional_square_sum
import copy

def kalman_log_likelihood(params, y_data, p, q):
    """
    Calcula el negativo de la log-verosimilitud exacta de un modelo ARMA
    usando el Filtro de Kalman.
    """
    phi = params[:p]
    theta = params[p:]
    n = len(y_data)
    
    r = max(p, q + 1)
    T = np.zeros((r, r))
    R = np.zeros((r, 1))
    H = np.zeros((1, r))
    Q = 1

    H[0, 0] = 1
    
    # --- CORRECCIÓN 1: Indexación de la matriz R ---
    R[0, 0] = 1 # El valor 1 va en la primera fila, primera columna.
    if q > 0:
        # El slicing aquí es correcto, asigna a la primera columna.
        R[1:q+1, 0] = theta

    if p > 0:
        T[0, :p] = phi
    if r > 1:
        T[1:r, :r-1] = np.eye(r-1)

    # Inicialización del Filtro
    z_hat = np.zeros((r, 1))
    sum_sq_std_errors = 0.0
    try:
        V = (R @ R.T) * Q
        # Resolver la ecuación de Lyapunov discreta para la covarianza incondicional P
        P = solve_discrete_lyapunov(T, V)
    except np.linalg.LinAlgError:
        # Si el proceso no es estacionario, la matriz puede ser singular. Usar una matriz identidad.
        P = np.eye(r)

    log_likelihood = 0.0
    
    # Bucle del Filtro de Kalman
    for t in range(n):
        v = y_data[t] - H @ z_hat
        F = H @ P @ H.T
        # Evitar división por cero si F es muy pequeño
        if F < 1e-9: 
            continue

        K = P @ H.T / F
        
        # Actualización del estado y la covarianza (forma estándar)
        z_hat_updated = z_hat + K @ v
        P_updated = P - K @ H @ P

        # Predicción para el siguiente paso
        z_hat = T @ z_hat_updated
        P = T @ P_updated @ T.T + (R @ R.T) * Q

        # El filtro de Kalman asume sigma=1. La verosimilitud se concentra.
        log_likelihood += -0.5 * (np.log(F) + (v**2 / F))
        sum_sq_std_errors += v**2 / F
        
    # Devolvemos el negativo de la log-verosimilitud para la minimización.
    # Se omite el término constante -n/2 * log(2*pi) ya que no afecta la optimización.
    return -log_likelihood.item(), sum_sq_std_errors.item()


def maximum_likelihood(model, y_data):
    """
    Ajusta un modelo ARMA(p,q) usando Máxima Verosimilitud Exacta.
    """
    model_copy = copy.deepcopy(model)

    p = model_copy.p
    q = model_copy.q

    # Usar los resultados de CSS como punto de partida (opcional pero recomendado)
    result_css = conditional_square_sum(model, y_data) # Tu función CSS
    
    phi_css = result_css.phi
    theta_css = result_css.theta
    initial_params = np.concatenate([phi_css, theta_css])
    
    # Wrapper para que la función objetivo devuelva solo el primer valor (log-likelihood)
    def objective_for_minimize(params, y_data, p, q):
        neg_log_lik, _ = kalman_log_likelihood(params, y_data, p, q)
        return neg_log_lik

    # Llamar al optimizador con la función de verosimilitud de Kalman
    result = minimize(objective_for_minimize, 
                      initial_params, 
                      args=(y_data, p, q),
                      method='BFGS')

    # Una vez encontrados los parámetros óptimos, actualizar el modelo
    phi_mle = result.x[:p]
    theta_mle = result.x[p:]
    
    # Calcular la varianza del error (sigma^2) con los parámetros MLE
    # Re-ejecutar el filtro con los parámetros óptimos para obtener la suma de errores al cuadrado
    _, sum_sq_std_errors = kalman_log_likelihood(result.x, y_data, p, q)

    # Calcular la varianza del error (sigma^2) y la desviación estándar (sigma)
    n = len(y_data)
    sigma2_mle = sum_sq_std_errors / n
    model_copy.ar.phi = phi_mle
    model_copy.ma.theta = theta_mle
    model_copy.sigma = np.sqrt(sigma2_mle)

    return model_copy