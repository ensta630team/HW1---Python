# source/optimize/parallel_bootstrap.py

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import warnings

from source.utils.sampling import get_windows
from source.optimize.hrissanen import HannanRissanen
from source.optimize.likelihood import maximum_likelihood_estimation

def _bootstrap_iteration(y_series, j_max, criteria, ols_on=False):
    """
    Performs a single iteration of the Hannan-Rissanen order selection.

    This is a private helper function designed to be called in parallel by joblib.
    It takes a single time series, finds the best ARMA(p,q) orders, and
    returns the selected orders and the initially fitted model.

    Args:
        y_series (np.ndarray): The time series data for this iteration.
        j_max (int): The maximum lag order to consider.
        criteria (str): The information criterion to use ('aic', 'bic', 'hqic').

    Returns:
        tuple: A tuple containing (best_p, best_q, initial_model). Returns
               (None, None, None) if the estimation fails.
    """
    try:
        p, q, initial_model, _ = HannanRissanen(y_series, 
                                                j_max=j_max, 
                                                criteria=criteria, 
                                                verbose=False,
                                                fran=ols_on)
        return p, q, initial_model
    except Exception:
        # If any error occurs during the estimation for a specific sample,
        # return None so it can be filtered out later.
        return None, None, None

def parallel_bootstrap_hr(y_data, 
                          iterations=100, 
                          j_max=20, 
                          sampling='windows',
                          model_generator=None, 
                          criteria='bic', 
                          ols_on=False,
                          n_jobs=-1):
    """
    Finds the most stable ARMA(p,q) orders using a parallelized bootstrap.

    This function evaluates the stability of the Hannan-Rissanen order selection
    procedure by running it on multiple resampled or simulated time series.
    It then identifies the most frequently selected (p,q) pair and refits
    the final model on the original data.

    Args:
        y_data (np.ndarray): The original time series data.
        iterations (int, optional): The number of bootstrap iterations to run.
            Defaults to 100.
        j_max (int, optional): The maximum AR and MA order to test in each
            iteration. Defaults to 20.
        parametric (bool, optional): If True, performs a parametric bootstrap by
            fitting a base model to `y_data` and simulating new series from it.
            If False, performs a non-parametric moving block bootstrap on `y_data`.
            Defaults to True.
        criteria (str, optional): The information criterion for order selection
            ('aic', 'bic', 'hqic'). Defaults to 'bic'.
        n_jobs (int, optional): The number of CPU cores to use for parallel
            processing. -1 means using all available cores. Defaults to -1.

    Returns:
        dict or None: A dictionary containing the results:
            - 'best_p' (int): The most frequently selected AR order.
            - 'best_q' (int): The most frequently selected MA order.
            - 'model' (ARMA): The final model, refitted on the original data.
            - 'param_distribution' (pd.DataFrame): A DataFrame showing the
              frequency and probability of each (p,q) pair found.
            Returns None if all bootstrap iterations fail.
    """
    print(f"Starting {iterations}-iteration parallel bootstrap...")
    n_obs = len(y_data)
    bootstrap_series = []
    if model_generator is not None:
        # DGP given
        print("Using given DGP to generate bootstrap series...")
        for _ in range(iterations):
            bootstrap_series.append(model_generator.sample(n_samples=n_obs, burn_in=100))
    elif sampling=='parametric':
        # Parametric Bootstrap: Fit a base model once and sample from it.
        print("Fitting base model to original data for parametric sampling...")
        base_p, base_q, base_model, _ = HannanRissanen(y_data, j_max=j_max, criteria=criteria, verbose=True)
        print(f"\nBase model found: ARMA({base_p},{base_q}). Generating bootstrap series...")
        for _ in range(iterations):
            bootstrap_series.append(base_model.sample(n_samples=n_obs, burn_in=100))
    else:
        # Non-parametric Bootstrap: Resample blocks from the original series.
        print("Generating non-parametric block bootstrap series...")
        bootstrap_series = get_windows(y_data, 
                                       n_ventanas=iterations,
                                       tamano_ventana=n_obs - n_obs // 4)

    # --- Run Hannan-Rissanen in parallel on all bootstrap series ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        results = Parallel(n_jobs=n_jobs)(
            delayed(_bootstrap_iteration)(series, j_max, criteria, ols_on) 
            for series in tqdm(bootstrap_series, desc="Processing bootstrap iterations")
        )
    
    # --- Process the results ---
    # Filter out any iterations that failed
    valid_results = [res for res in results if res[0] is not None]
    if not valid_results:
        print("Warning: All bootstrap iterations failed to converge.")
        return None

    found_params = [(p, q) for p, q, _ in valid_results]
    
    # Store the first successfully fitted model for each (p,q) pair
    first_model_for_param = {}
    for p, q, model in valid_results:
        if (p, q) not in first_model_for_param:
            first_model_for_param[(p, q)] = model

    # Find the most frequent (p,q) pair (the mode)
    param_counts = pd.Series(found_params).value_counts()
    most_frequent_params = param_counts.index[0]
    best_p, best_q = most_frequent_params

    print(f"\nBootstrap complete! Most stable pair: ({best_p},{best_q}) "
          f"[{param_counts.iloc[0]}/{len(valid_results)} successful iterations]")
    
    # Refit the final model on the original data using the best (p,q) structure
    print("Refitting the final model on the original data using MLE...")
    model_to_refit = first_model_for_param[most_frequent_params]
    best_model = maximum_likelihood_estimation(model_to_refit, y_data)
    
    # Create a summary DataFrame of the parameter distribution
    param_distribution = param_counts.reset_index()
    param_distribution.columns = ['(p,q)', 'frequency']
    param_distribution['probability'] = param_distribution['frequency'] / len(valid_results)
    
    return {
        'best_p': best_p,
        'best_q': best_q,
        'model': best_model,
        'param_distribution': param_distribution
    }