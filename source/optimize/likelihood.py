# likelihood.py
import numpy as np
import copy
from scipy.optimize import minimize

def _objective_function_mle(params, y_data, p, q):
    """
    Private helper function to calculate the negative log-likelihood for an ARMA(p,q) model.

    This function is minimized by scipy.optimize.minimize to find the Maximum
    Likelihood Estimates (MLE) of the model parameters. The parameters are
    packed into a single array for the optimizer.

    Args:
        params (np.ndarray): A 1D array containing the model parameters in the
            order [constant (c), phi_1, ..., phi_p, theta_1, ..., theta_q].
        y_data (np.ndarray): The time series data.
        p (int): The order of the AR component.
        q (int): The order of the MA component.

    Returns:
        float: The negative of the concentrated log-likelihood value. Returns
               np.inf if the parameters lead to a non-finite result.
    """
    # Unpack parameters
    c = params[0]
    phi = params[1 : 1 + p]
    theta = params[1 + p :]
    
    n = len(y_data)
    errors = np.zeros(n)
    start_t = max(p, q)
    
    # Recursively compute the one-step-ahead prediction errors (residuals)
    for t in range(start_t, n):
        y_past = y_data[t-p:t][::-1]
        u_past = errors[t-q:t][::-1]
        
        ar_part = np.dot(phi, y_past) if p > 0 else 0
        ma_part = np.dot(theta, u_past) if q > 0 else 0
        
        errors[t] = y_data[t] - c - ar_part - ma_part

    # Calculate the sum of squared residuals (SSR)
    ssr = np.sum(errors[start_t:]**2)
    
    # Handle cases where the optimization explores unstable regions
    if not np.isfinite(ssr):
        return np.inf

    # Calculate the concentrated log-likelihood
    effective_n = n - start_t
    if effective_n <= 0 or ssr <= 1e-12: # Avoid division by zero or log(0)
        return np.inf

    sigma2 = ssr / effective_n
    
    # The formula for the concentrated log-likelihood of a Gaussian ARMA model
    log_likelihood = -effective_n / 2 * (np.log(2 * np.pi) + np.log(sigma2) + 1)

    if not np.isfinite(log_likelihood):
        return np.inf

    # The optimizer minimizes, so we return the negative log-likelihood
    return -log_likelihood


def _objective_function_css(params, y_data, p, q):
    """
    Private helper function to calculate the Conditional Sum of Squares (CSS).

    This serves as a more stable objective function to get good initial
    parameter estimates before running the full MLE.

    Args:
        params (np.ndarray): Parameter vector [c, phis, thetas].
        y_data (np.ndarray): The time series data.
        p (int): The AR order.
        q (int): The MA order.

    Returns:
        float: The conditional sum of squared residuals.
    """
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


def maximum_likelihood_estimation(model, y_data, return_likelihood=False):
    """
    Fits an ARMA model using a two-step Conditional Sum of Squares-Maximum
    Likelihood Estimation (CSS-MLE) strategy.

    This approach is robust against poor initial values. It first finds
    preliminary estimates using CSS and then refines them using the full MLE.

    Args:
        model (ARMA): An instance of the ARMA class with the desired (p,q) orders.
        y_data (np.ndarray): The time series data.
        return_likelihood (bool, optional): If True, returns the final
            log-likelihood value along with the fitted model. Defaults to False.

    Returns:
        ARMA or tuple:
            - If return_likelihood is False: The fitted ARMA model instance.
            - If return_likelihood is True: A tuple containing (fitted_model, log_likelihood).
    """
    with np.errstate(over='ignore', invalid='ignore'):
        model_copy = copy.deepcopy(model)
        n = len(y_data)
        p = model_copy.p
        q = model_copy.q

        # --- Step 1: Get robust initial parameter estimates using CSS ---
        # Start with simple initial values: c=mean, phis=0, thetas=0
        css_initial_params = np.concatenate(([np.mean(y_data)], np.zeros(p + q)))
        
        css_result = minimize(
            _objective_function_css,
            css_initial_params,
            args=(y_data, p, q),
            method='Nelder-Mead', # A gradient-free method, good for less smooth surfaces
            options={'maxiter': 5000}
        )
        
        # The result from CSS becomes the starting point for MLE
        mle_initial_params = css_result.x
        
        # --- Step 2: Refine the estimates using full MLE ---
        result = minimize(
            _objective_function_mle,
            mle_initial_params,
            args=(y_data, p, q),
            method='BFGS' # A gradient-based method, efficient with good initial values
        )

        # --- Extract final parameters and compute final sigma ---
        optimal_params = result.x
        log_likelihood = -result.fun

        start_t = max(p, q)
        effective_n = n - start_t
        
        if effective_n > 0:
            # The MLE for sigma^2 can be backed out from the final log-likelihood
            log_sigma2 = -2 * log_likelihood / effective_n - np.log(2 * np.pi) - 1
            error_variance = np.exp(log_sigma2)
            estimated_sigma = np.sqrt(error_variance)
        else:
            estimated_sigma = np.nan

        # --- Update the model instance with the final fitted parameters ---
        model_copy.c = optimal_params[0]
        model_copy.ar.c = optimal_params[0]
        model_copy.ma.c = optimal_params[0]

        model_copy.ar.phi = optimal_params[1 : 1 + p]
        model_copy.ma.theta = optimal_params[1 + p :]
        model_copy.sigma = estimated_sigma
        
        if return_likelihood:
            return model_copy, log_likelihood
        else:
            return model_copy