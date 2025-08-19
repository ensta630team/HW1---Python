import numpy as np 
from tqdm import tqdm
import pandas as pd
from source.models.ols import OLS
from source.models.arma import ARMA
from source.optimize.likelihood import maximum_likelihood_estimation

def HannanRissanen(y_data, j_max=20, criteria='bic', verbose=False, fran=False):
    """
    Implements the 3-step Hannan-Rissanen algorithm for ARMA(p,q) order selection.

    This method provides a computationally efficient way to find good candidate
    orders (p,q) for an ARMA model by avoiding a full Maximum Likelihood (MLE)
    grid search.

    The steps are:
    1. Fit a long autoregressive model (AR(j_max)) to the data using OLS
       to obtain estimates of the residuals (errors).
    2. Perform a grid search over (p,q) up to j_max. For each pair, fit an
       OLS regression using lagged values of the series (for the AR part) and
       lagged residuals from step 1 (as proxies for the MA part).
    3. Select the best (p,q) pair based on an information criterion (AIC, BIC,
       or HQIC) calculated from the OLS residuals of step 2.
    4. Refit the final ARMA(p,q) model using full MLE, starting with the
       OLS estimates as initial values.

    Args:
        y_data (np.ndarray): The univariate time series data.
        j_max (int, optional): The maximum lag order to consider for p, q, and
            for the initial long AR model. Defaults to 20.
        criteria (str, optional): The information criterion to use for model
            selection. Can be 'aic', 'bic', or 'hqic'. Defaults to 'bic'.
        verbose (bool, optional): If True, displays progress bars and final
            model information. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - best_p (int): The selected autoregressive order.
            - best_q (int): The selected moving average order.
            - final_model (ARMA): The final ARMA(p,q) model instance, fitted
              using MLE.
            - lvalue (float): The log-likelihood value of the final fitted model.
    """
    n = len(y_data)
    if n < 2 * j_max:
        raise ValueError("Time series is too short for the chosen j_max.")

    # --- Step 1: Fit a long AR(j_max) model to get residual estimates ---
    X = np.array([y_data[i : i + j_max] for i in range(n - j_max)])
    y = y_data[j_max:]
    
    ols_step1 = OLS()
    ols_step1.fit(X, y)
    u_hat = y - ols_step1.predict(X)

    # --- Step 2: Grid search over (p,q) using OLS ---
    # Prepare regressors for the second stage regression
    y_target = y_data[j_max:]
    
    # AR regressors: y_{t-1}, ..., y_{t-p}
    X_ar_full = np.array([y_data[j_max - i - 1 : n - i - 1] for i in range(j_max)]).T
    
    # MA regressors (proxies): û_{t-1}, ..., û_{t-q}
    # Pad residuals to align with the target variable y_target
    u_hat_padded = np.concatenate([np.zeros(j_max), u_hat])
    X_ma_full = np.array([u_hat_padded[j_max - i - 1 : n - i - 1] for i in range(j_max)]).T
    param_combinations = [(p, q) for p in range(j_max + 1) for q in range(j_max + 1) if p != 0 or q != 0]
    
    results = []
    best_value = np.inf
    best_params = (None, None)
    best_ols_model = None

    pbar_desc = f"Searching for best (p,q) via {criteria.upper()} [Fast]"
    pbar = tqdm(param_combinations, total=len(param_combinations), disable=not verbose, desc=pbar_desc)
    
    for p, q in pbar:
        # Construct the regressor matrix for the current ARMA(p,q) model
        X_ar_p = X_ar_full[:, :p] if p > 0 else np.empty((len(y_target), 0))
        X_ma_q = X_ma_full[:, :q] if q > 0 else np.empty((len(y_target), 0))
        X_arma = np.concatenate([X_ar_p, X_ma_q], axis=1)
        
        # Fit the second-stage OLS regression
        ols_step2 = OLS()
        ols_step2.fit(X_arma, y_target)
        
        # Calculate information criteria based on the residual sum of squares (RSS)
        residuals = y_target - ols_step2.predict(X_arma)
        rss = np.sum(residuals**2)
        
        N = len(y_target)
        k = p + q + 1  # Number of parameters in the OLS regression (p + q + intercept)

        if rss <= 1e-12:
            current_value = -np.inf
        else:
            log_rss_N = np.log(rss / N)
            if criteria == 'bic':
                current_value = log_rss_N + (k * np.log(N)) / N
            elif criteria == 'hqic':
                current_value = log_rss_N + (2 * k * np.log(np.log(N))) / N
            else:  # Default to 'aic'
                current_value = log_rss_N + (2 * k) / N
            
        results.append({'p': p, 'q': q, criteria: current_value})

        # Update the best model if a lower criterion value is found
        if current_value < best_value:
            best_value = current_value
            best_params = (p, q)
            best_ols_model = ols_step2
    
    best_p, best_q = best_params
    if verbose:
        print(f"\nFast search complete! Best OLS Model: ARMA({best_p},{best_q}). Refining with MLE...")

    if fran: # idea del fran
        final_models = []
        final_likeli = []
        for i in range(3):
            final_arma_model = ARMA(c=0., 
                                    phi_params='stationary', 
                                    theta_params='stationary', 
                                    p=best_p, q=best_q)
            
            final_model, lvalue = maximum_likelihood_estimation(final_arma_model, y_data, return_likelihood=True)
            final_models.append(final_model)
            final_likeli.append(lvalue)

        best_final_model = final_models[np.argmin(final_likeli)]
        best_final_likelihood = np.min(final_likeli)

        return best_p, best_q, best_final_model, best_final_likelihood
    else:
        phi_init = best_ols_model.coef_[:best_p]
        theta_init = best_ols_model.coef_[best_p:]
        c_init = best_ols_model.intercept_

        final_arma_model = ARMA(c=c_init, phi_params=phi_init, theta_params=theta_init)
        
        final_model, lvalue = maximum_likelihood_estimation(final_arma_model, y_data, return_likelihood=True)

        return best_p, best_q, final_model, lvalue