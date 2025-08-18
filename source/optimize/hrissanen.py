# source/optimize/hrissanen.py
import numpy as np
from tqdm import tqdm
import pandas as pd
from source.models.ols import OLS
from source.models.arma import ARMA
from source.optimize.likelihood import maximum_likelihood_estimation

def HannanRissanen(y_data, j_max=20, criteria='bic', verbose=False):
    n = len(y_data)
    X = np.array([y_data[i : i + j_max] for i in range(n - j_max)])
    y = y_data[j_max:]
    ols_step1 = OLS(); ols_step1.fit(X, y); u_hat = y - ols_step1.predict(X)
    y_target = y_data[j_max:]
    X_ar_full = np.array([y_data[j_max - i - 1 : n - i - 1] for i in range(j_max)]).T
    u_hat_padded = np.concatenate([np.zeros(j_max), u_hat])
    X_ma_full = np.array([u_hat_padded[j_max - i - 1 : n - i - 1] for i in range(j_max)]).T
    param_combinations = [(p, q) for p in range(j_max + 1) for q in range(j_max + 1) if p != 0 or q != 0]
    results = []; best_value = np.inf; best_params = (None, None)
    
    pbar = tqdm(param_combinations, total=len(param_combinations), disable=not verbose, desc="Buscando (p,q) [Rápido]")
    
    for p, q in pbar:
        X_ar_p = X_ar_full[:, :p] if p > 0 else np.empty((len(y_target), 0))
        X_ma_q = X_ma_full[:, :q] if q > 0 else np.empty((len(y_target), 0))
        X_arma = np.concatenate([X_ar_p, X_ma_q], axis=1)
        ols_step2 = OLS(); ols_step2.fit(X_arma, y_target)
        residuals = y_target - ols_step2.predict(X_arma)
        rss = np.sum(residuals**2)
        N = len(y_target); k = p + q + 1 
        if rss <= 1e-12: # Si el ajuste es perfecto, evitamos log(0)
            current_value = -np.inf
        else:
            log_rss_N = np.log(rss / N)
            
            # --- LÓGICA ACTUALIZADA PARA MANEJAR LOS TRES CRITERIOS ---
            if criteria == 'bic':
                current_value = log_rss_N + (k * np.log(N)) / N
            elif criteria == 'hqic':
                current_value = log_rss_N + (2 * k * np.log(np.log(N))) / N
            else: # 'aic' es el predeterminado
                current_value = log_rss_N + (2 * k) / N
            
        results.append({'p': p, 'q': q, criteria: current_value})
        if current_value < best_value:
            best_value = current_value
            best_params = (p, q)
            best_ols_model = ols_step2
    
    best_p, best_q = best_params
    if verbose: print(f"\nMejor Modelo OLS: ARMA({best_p},{best_q}). Refinando con MLE...")
    phi_init = best_ols_model.coef_[:best_p]
    theta_init = best_ols_model.coef_[best_p:]
    c_init = best_ols_model.intercept_
    final_arma_model = ARMA(c=c_init, phi_params=phi_init, theta_params=theta_init)
    final_model, lvalue = maximum_likelihood_estimation(final_arma_model, y_data, return_likelihood=True)
    return best_p, best_q, final_model, lvalue