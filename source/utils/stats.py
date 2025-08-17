import numpy as np 
import sympy as sp

def calculate_irf(F, H, phi=None, method='exact'):
    p = F.shape[-1]

    irf_values = np.zeros(H)
    irf_values[0] = 1

    eigenvals, eigenvec = np.linalg.eig(F)
    
    # check eigenvalues (we can use schur function from scipy CHECK!)
    mult = len(set(eigenvals))
    if len(eigenvals) != mult  or method=='jordan':
        Fsymp = sp.Matrix(F)
        P, J = Fsymp.jordan_form()
        print(J)

    elif method == 'exact' and len(eigenvals) == mult:
        for h in range(1, H):
            temp = np.eye(p)*np.pow(eigenvals, h)
            temp = np.matmul(eigenvec, temp)
            temp = np.matmul(temp, np.linalg.inv(eigenvec))
            irf_values[h] = temp[0, 0]

    elif method == 'exact_old' and len(eigenvals) == mult:
        for h in range(1, H):
            F_h = np.linalg.matrix_power(F, h)
            irf_values[h] = F_h[0, 0]

    if method == 'simulation':
        for h in range(1, H):
            # Get the last p values of the IRF generated so far
            past_values = irf_values[max(0, h - F.shape[-1]):h]
            # The coefficients to use depend on how many past values we have
            coeffs_to_use = phi[:len(past_values)]
            irf_values[h] = np.dot(coeffs_to_use, past_values[::-1])
    
    return irf_values

def calculate_acf(data: np.ndarray, lags: int) -> np.ndarray:
    """
    Calcula la Función de Autocorrelación (ACF) para una serie de tiempo.

    Args:
        data (np.ndarray): La serie de tiempo (1D array).
        lags (int): El número máximo de rezagos para calcular.

    Returns:
        np.ndarray: Un array con los valores de la ACF para los rezagos 0 a `lags`.
    """
    n = len(data)
    if n <= lags:
        raise ValueError("El número de lags debe ser menor que la longitud de los datos.")

    acf_values = np.zeros(lags + 1)
    mean = np.mean(data)
    variance = np.sum((data - mean)**2)

    if variance == 0: # Evitar división por cero para series constantes
        return np.zeros(lags + 1)

    acf_values[0] = 1.0

    for k in range(1, lags + 1):
        # Numerador: sum((y_t - mean) * (y_{t-k} - mean))
        numerator = np.sum((data[k:] - mean) * (data[:-k] - mean))
        acf_values[k] = numerator / variance

    return acf_values

def calculate_pacf(data: np.ndarray, lags: int) -> np.ndarray:
    """
    Calcula la Función de Autocorrelación Parcial (PACF) para una serie de tiempo.
    Usa el método de resolución de ecuaciones de Yule-Walker para cada lag.

    Args:
        data (np.ndarray): La serie de tiempo (1D array).
        lags (int): El número máximo de rezagos para calcular.

    Returns:
        np.ndarray: Un array con los valores de la PACF para los rezagos 0 a `lags`.
                    El valor para el rezago 0 es NaN por convención.
    """
    n = len(data)
    if n <= lags:
        raise ValueError("El número de lags debe ser menor que la longitud de los datos.")

    acf_values = calculate_acf(data, lags)
    pacf_values = np.zeros(lags + 1)
    
    # Por convención, PACF en lag 0 es indefinido o 1, pero no se suele graficar.
    # Usaremos NaN para el lag 0 y 1 para el plot_acf_pacf, pero el cálculo empieza en 1.
    pacf_values[0] = np.nan 

    if lags >= 1:
        # PACF en el rezago 1 es igual a ACF en el rezago 1
        pacf_values[1] = acf_values[1]

    # Calcular PACF para rezagos k = 2 a `lags`
    for k in range(2, lags + 1):
        # Matriz R para el sistema de Yule-Walker
        # R es una matriz de autocorrelaciones (k x k)
        # R_ij = acf_values[abs(i - j)]
        R = np.zeros((k, k))
        for i in range(k):
            for j in range(k):
                R[i, j] = acf_values[abs(i - j)]
        
        # Vector r de autocorrelaciones (k x 1)
        # r_i = acf_values[i+1]
        r = acf_values[1 : k + 1]

        # Resolver R * phi = r para phi (los coeficientes AR de un modelo AR(k))
        try:
            phi_coeffs = np.linalg.solve(R, r)
            # La PACF en el rezago k es el último coeficiente del modelo AR(k)
            pacf_values[k] = phi_coeffs[-1]
        except np.linalg.LinAlgError:
            # En caso de una matriz singular (raro pero posible con datos degenerados)
            pacf_values[k] = np.nan
            print(f"Advertencia: No se pudo calcular PACF para el rezago {k} debido a una matriz singular.")

    return pacf_values