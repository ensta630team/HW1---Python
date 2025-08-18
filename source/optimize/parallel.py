# source/optimize/parallel_bootstrap.py
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from source.utils.sampling import get_windows
from source.optimize.hrissanen import HannanRissanen
from source.optimize.likelihood import maximum_likelihood_estimation
import warnings

def _bootstrap_iteration(y_series, j_max, criteria, model):
    y_step = model.sample(n_samples=len(y_series), burn_in=100) if model else y_series
    p, q, final_model, _ = HannanRissanen(y_step, j_max=j_max, criteria=criteria, verbose=False)
    return p, q, final_model

def parallel_bootstrap_hr(y_data, iterations=100, j_max=20, model=None, criteria='bic', n_jobs=-1):
    print(f"Iniciando Bootstrap con {iterations} iteraciones...")
    if model is None:
        nobs = y_data.shape[0]
        y_data_bs = get_windows(y_data, n_ventanas=iterations, tamano_ventana=nobs - nobs // 4)
    else:
        y_data_bs = [y_data] * iterations
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        results = Parallel(n_jobs=n_jobs)(
            delayed(_bootstrap_iteration)(y, j_max, criteria, model) 
            for y in tqdm(y_data_bs, desc="Procesando iteraciones")
        )
    
    found_params = [(p, q) for p, q, _ in results]
    first_model_for_param = { (p, q): m for p, q, m in results if (p, q) not in locals().get('first_model_for_param', {}) }

    param_counts = pd.Series(found_params).value_counts()
    if param_counts.empty:
        print("Advertencia: Ninguna iteración convergió.")
        return None
        
    most_frequent_params = param_counts.index[0]
    best_p, best_q = most_frequent_params
    print(f"\nPar más estable: ({best_p},{best_q}) [{param_counts.iloc[0]}/{iterations} veces]")
    
    model_to_refit = first_model_for_param[most_frequent_params]
    print("Reajustando el modelo final a los datos originales...")
    best_model = maximum_likelihood_estimation(model_to_refit, y_data)
    
    param_distribution = param_counts.reset_index()
    param_distribution.columns = ['(p,q)', 'frequency']
    param_distribution['probability'] = param_distribution['frequency'] / iterations
    
    return {'best_p': best_p, 'best_q': best_q, 'model': best_model, 'param_distribution': param_distribution}