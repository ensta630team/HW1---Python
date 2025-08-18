import numpy as np 
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from source.models.arma import ARMA
from source.optimize.hrissanen import HannanRissanen

def parametric_bootstrap_hr(y_data, 
                            iterations=100, 
                            j_max=20, 
                            criteria='hqic',
                            n_jobs=-1):
    """
    Evalúa la estabilidad de la selección de órdenes (p,q) de Hannan-Rissanen
    utilizando un bootstrap paramétrico.

    Args:
        y_data (np.ndarray): La serie de tiempo original.
        iterations (int): Número de iteraciones del bootstrap.
        j_max (int): Orden máximo a probar en cada iteración.
        criteria (str): Criterio de información ('aic', 'bic', 'hqic').
        n_jobs (int): Número de procesos a usar en paralelo (-1 para todos).

    Returns:
        dict: Un diccionario con los resultados del bootstrap.
    """
    n_obs = len(y_data)

    # --- PASO 1: Ajustar un modelo base a los datos originales ---
    base_p, base_q, base_model, _ = HannanRissanen(y_data, 
                                                   j_max=j_max, 
                                                   criteria=criteria, 
                                                   verbose=True)

    # --- PASO 2: Generar 'iterations' series de tiempo a partir del modelo base ---
    bootstrap_series = []
    for _ in tqdm(range(iterations), desc="Generando series bootstrap"):
        y_step = base_model.sample(n_samples=n_obs, burn_in=100)
        bootstrap_series.append(y_step)

    # --- PASO 3: Ejecutar Hannan-Rissanen en paralelo para cada serie bootstrap ---
    def _hr_iteration(series):
        p, q, _, _ = HannanRissanen(series, j_max=j_max, criteria=criteria, verbose=False)
        return (p, q)

    results = Parallel(n_jobs=n_jobs)(
        delayed(_hr_iteration)(y_series) 
        for y_series in tqdm(bootstrap_series, desc="Procesando iteraciones")
    )

    # --- PASO 4: Analizar los resultados ---
    param_counts = pd.Series(results).value_counts()
    
    if param_counts.empty:
        print("Advertencia: Ninguna iteración del bootstrap convergió.")
        return None
        
    most_frequent_params = param_counts.index[0]
    best_p, best_q = most_frequent_params

    print(f"\n¡Bootstrap completado! El par más estable es p={best_p}, q={best_q} (apareció {param_counts.iloc[0]} de {iterations} veces).")

    # (Opcional) Crear un DataFrame con las frecuencias para un análisis más detallado
    freq_df = param_counts.reset_index()
    freq_df.columns = ['(p,q)', 'frequency']
    freq_df['probability'] = freq_df['frequency'] / iterations

    return {
        'best_p': best_p,
        'best_q': best_q,
        'base_model': base_model, # El modelo ajustado a los datos originales
        'param_distribution': freq_df # La distribución de los órdenes encontrados
    }