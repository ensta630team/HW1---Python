import numpy as np 
from source.models.ols import OLS
from source.models.arma import ARMA
from source.data.preprocessing import to_X_y
from source.optimize.likelihood import maximum_likelihood_estimation
from tqdm import tqdm
import matplotlib.pyplot as plt 
import pandas as pd

def HannanRissanen(y_data, j_max=20, criteria='hqic'):
    n = len(y_data)

    X, y = to_X_y(y_data, j_max)
    X = X[:, 1:]

    ols_model = OLS()
    ols_model.fit(X, y)
    y_pred = ols_model.predict(X)
    
    # residuals
    u_hat = y - y_pred

    U_h, y_h = to_X_y(u_hat, j_max)
    U_h = U_h[:, 1:]

    y_target = y[j_max:]
    X_reg = X[j_max:, :]
    X_step2_full = np.concatenate([X_reg, U_h], axis=1) 

    param_combinations = [
        (p, q) for p in range(j_max + 1) for q in range(j_max + 1) if p != 0 or q != 0
    ]

    results = []
    best_value = np.inf
    best_params = (None, None)

    pbar = tqdm(param_combinations, 
                total=len(param_combinations), 
                desc="Looking for the best (p,q)")

    for p, q in pbar:
        X_ar = X_step2_full[:, :p] if p > 0 else np.empty((len(y_target), 0))
        X_ma = X_step2_full[:, j_max : j_max + q] if q > 0 else np.empty((len(y_target), 0))
        X_arma = np.concatenate([X_ar, X_ma], axis=1)
        ols_curr = OLS()
        ols_curr.fit(X_arma, y_target)

        y_pred_step2 = ols_curr.predict(X_arma)
        residuals_step2 = y_target - y_pred_step2
        rss = np.sum(residuals_step2**2)

        N = len(y_target)
        k = p + q + 1

        if rss <= 1e-9:
            bic = -np.inf
        else:
            aic = np.log(rss/N) + 2 * k/N
            bic = np.log(rss/N) + k/N * np.log(N)
            hqic = np.log(rss/N) + 2 * k/N * np.log(np.log(N))
                    
        results.append({'p': p, 'q': q, 'bic': bic, 'aic': aic, 'hqic': hqic})

        # Actualizar el mejor modelo si se encuentra un BIC más bajo
        if results[-1][criteria] < best_value:
            best_value = results[-1][criteria]
            best_params = (p, q)

    best_p, best_q = best_params
    print(f"\nSearching Done! Best Model: p={best_p}, q={best_q} - {criteria}={best_value:.4f}")

    # Ajustar el modelo final con los mejores parámetros encontrados
    X_ar_final = X_step2_full[:, :best_p] if best_p > 0 else np.empty((len(y_target), 0))
    X_ma_final = X_step2_full[:, j_max : j_max + best_q] if best_q > 0 else np.empty((len(y_target), 0))
    X_arma_final = np.concatenate([X_ar_final, X_ma_final], axis=1)

    final_model = OLS()
    final_model.fit(X_arma_final, y_target)
    
    arma_final = ARMA(c=final_model.intercept_, 
                      phi_params=final_model.coef_[:best_p], 
                      theta_params=final_model.coef_[best_p:])

    arma_final = maximum_likelihood_estimation(arma_final, y_data)

    results_df = pd.DataFrame(results).set_index(['p', 'q'])
    return best_p, best_q, arma_final, results_df