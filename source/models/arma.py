# HW1/source/models/arma.py

import pandas as pd
import numpy as np

from source.models.base import TimeSeriesModel
from source.optimize.likelihood import maximum_likelihood_estimation
from source.models.ar import AutoRegressive
from source.models.ma import MovingAverage
from tqdm import tqdm
import warnings



class ARMA(TimeSeriesModel):
    """
    Implements an Autoregressive Moving Average (ARMA) model.

    This class defines an ARMA(p,q) process, combining both AR and MA components.

    The process is defined as:
    y_t = c + φ₁*y_{t-1} + ... + φp*y_{t-p} + u_t + θ₁*u_{t-1} + ... + θq*u_{t-q}
    """
    def __init__(self,
                 c: float = 0.0,
                 sigma: float = 1.0,
                 phi_params=None,
                 theta_params=None,
                 **kwargs):
        """
        Initializes the ARMA model.

        Args:
            c (float): The constant term of the model.
            sigma (float): The standard deviation of the white noise error term.
            phi_params (list or np.ndarray, optional): Vector of AR parameters (φ coefficients).
            theta_params (list or np.ndarray, optional): Vector of MA parameters (θ coefficients).
        """
        # ==== AR Setup ====
        self.ar = AutoRegressive(c, sigma, phi_params, **kwargs)

        # ==== MA Setup ====
        self.ma = MovingAverage(c, sigma, theta_params, **kwargs)
        
        super().__init__(c, sigma)

    @property
    def params(self) -> int:
        """The order (p) of the AR component."""
        return {'phi': self.ar.phi, 'theta': self.ma.theta}
    
    @property
    def p(self) -> int:
        """The order (p) of the AR component."""
        return self.ar.p

    @property
    def q(self) -> int:
        """The order (q) of the MA component."""
        return self.ma.q
        
    @property
    def phi(self) -> np.ndarray:
        """The coefficients (phi) of the AR component."""
        return self.ar.phi

    @property
    def theta(self) -> np.ndarray:
        """The coefficients (theta) of the MA component."""
        return self.ma.theta
    
    def _is_stationary(self) -> bool:
        """
        Checks for stationarity. An ARMA(p,q) process is stationary if its
        AR component is stationary. The MA component is always stationary.
        """
        if self.ar.p == 0:
            return True # Pure MA is always stationary
        return self.ar._is_stationary()


    def get_unconditional_mean(self) -> float:
        """
        Calculates the unconditional mean of the ARMA process.
        The mean is determined only by the AR component and the constant.
        μ = c / (1 - φ₁ - φ₂ - ... - φp)
        """
        if not self._is_stationary():
            return np.nan
        
        phi_sum = np.sum(self.ar.phi)
        if np.isclose(phi_sum, 1.0):
            return np.inf  # Unit root case
            
        return self.c / (1 - phi_sum)

    def get_unconditional_std(self) -> float:
        """
        Calculates the unconditional standard deviation of the ARMA process.

        The variance is computed using the MA(inf) representation of the process:
        Var(y_t) = sigma^2 * sum(psi_j^2 for j=0 to inf),
        where psi_j are the coefficients of the impulse response function.
        """
        # Devolver el valor guardado en caché si ya se calculó
        if self._unconditional_std is not None:
            return self._unconditional_std

        # La desviación estándar incondicional solo está definida para procesos estacionarios
        if not self._is_stationary():
            self._unconditional_std = np.inf
            return self._unconditional_std

        # Caso 1: Proceso MA(q) puro (p=0). Usar la fórmula exacta y más simple.
        if self.p == 0:
            theta_sq_sum = np.sum(self.theta**2) if self.q > 0 else 0
            variance = self.sigma**2 * (1 + theta_sq_sum)
            self._unconditional_std = np.sqrt(variance)
            return self._unconditional_std

        # Caso 2: Proceso AR(p) puro (q=0). Usar el método exacto de Yule-Walker de la clase AR.
        if self.q == 0:
            gamma_0 = self.ar._solve_yule_walker_for_variance()
            self._unconditional_std = np.sqrt(gamma_0)
            return self._unconditional_std
            
        # Caso 3: Proceso ARMA(p,q) mixto. Usar la aproximación con la IRF.
        # Se necesita un horizonte suficientemente grande para que la suma de los psi^2 converja.
        # 1000 períodos es más que suficiente para la mayoría de los procesos estacionarios.
        H = 1000 

        # Obtener los coeficientes psi (los valores de la IRF)
        psi_coeffs = self.get_irf(H=H)

        # Calcular la varianza: sigma^2 * sum(psi_j^2)
        variance = self.sigma**2 * np.sum(psi_coeffs**2)

        # Guardar en caché el resultado y devolverlo
        self._unconditional_std = np.sqrt(variance)
        return self._unconditional_std

    def sample(self, n_samples: int, 
               initial_values: np.ndarray = None, 
               burn_in: int = 0, 
               **kwargs) -> np.ndarray:
        """
        Generates a time series sample from the ARMA(p,q) process.

        Args:
            n_samples (int): The desired length of the final time series.
            burn_in (int): The number of initial samples to discard.

        Returns:
            np.ndarray: The generated time series of length `n_samples`.
        """
        total_samples = n_samples + burn_in
        
        # Generate all shocks at once
        shocks = np.random.normal(loc=0, scale=self.sigma, size=total_samples)
        y_sample = np.zeros(total_samples)

        if initial_values is not None:
            if len(initial_values) != self.p:
                raise ValueError(f"initial_values must have length p={self.p}, but got {len(initial_values)}.")
            y_sample[:self.p] = initial_values

        # Determine the maximum lag needed for the loop start
        max_lag = max(self.ar.p, self.ma.q)

        # Iterate through time to build the series
        for t in range(max_lag, total_samples):
            # AR part: dot product of phi coefficients and past y values
            past_y = y_sample[t-self.ar.p:t]
            ar_component = np.dot(self.ar.phi, past_y[::-1]) if self.ar.p > 0 else 0
            
            # MA part: dot product of theta coefficients and past shocks
            past_shocks = shocks[t-self.ma.q:t]
            ma_component = np.dot(self.ma.theta, past_shocks[::-1]) if self.ma.q > 0 else 0
            
            # Combine all components
            y_sample[t] = self.c + ar_component + shocks[t] + ma_component
            
        return y_sample[burn_in:]
    
    def get_irf(self, H: int = 20) -> np.ndarray:
        """
        Calculates the Impulse Response Function (IRF) for an ARMA(p,q) process.

        The IRF is found by solving the recursive formula derived from the
        MA(∞) representation of the model.

        Args:
            H (int): The number of periods (horizon) for which to calculate the IRF.

        Returns:
            np.ndarray: An array of length H containing the IRF values.
        """
        phi_ext = np.zeros(H)
        theta_ext = np.zeros(H)
        
        phi_ext[1:self.ar.p + 1] = self.ar.phi
        theta_ext[1:self.ma.q + 1] = self.ma.theta

        irf_values = np.zeros(H)
        irf_values[0] = 1 

        for j in range(1, H):
            ar_part = np.dot(phi_ext[1:j + 1], irf_values[j-1::-1])

            ma_part = theta_ext[j]
            
            irf_values[j] = ar_part + ma_part
            
        return irf_values
    
    def __str__(self) -> str:
        summary_lines = self.ar.__str__() +'\n'+self.ma.__str__()
        return summary_lines
    


#### =========================================================
#### =================== ARMA GRID SEARCH ====================
#### =========================================================
class ARMAGridSearch:
    def __init__(self, data_series, p_max, q_max, criterion='aic'):
        """
        Inicializa la búsqueda en rejilla para modelos ARMA.

        Args:
            data_series (np.ndarray or pd.Series): La serie de tiempo a modelar.
            p_max (int): El orden máximo para el componente AR.
            q_max (int): El orden máximo para el componente MA.
            criterion (str): El criterio para seleccionar el mejor modelo ('aic' o 'bic').
        """
        self.data = np.asarray(data_series)
        self.p_max = p_max
        self.q_max = q_max
        self.criterion = criterion.lower()
        if self.criterion not in ['aic', 'bic']:
            raise ValueError("El criterio debe ser 'aic' o 'bic'.")
            
        self.results_ = None
        self.best_model_ = None

    def fit(self):
        """
        Ejecuta la búsqueda en rejilla, ajustando un modelo ARMA para cada
        combinación de (p,q) y calculando los criterios de información.
        """
        results_list = []
        n_obs = len(self.data)
        
        # 1. Generar todas las combinaciones de (p,q) excepto (0,0)
        param_combinations = [
            (p, q) for p in range(self.p_max + 1) for q in range(self.q_max + 1) if p != 0 or q != 0
        ]

        print(f"Iniciando Grid Search para {len(param_combinations)} combinaciones de ARMA(p,q)...")
        
        # 2. Iterar sobre las combinaciones con una barra de progreso tqdm
        progress_bar = tqdm(param_combinations, 
                            desc="Iniciando Grid Search", 
                            unit="modelo")
        
        for p, q in progress_bar:
            # 3. Actualizar la descripción de la barra de progreso en cada iteración
            progress_bar.set_description(f"Ajustando ARMA({p},{q})")
            try:
                # Crear un modelo base para ajustar
                model_to_fit = ARMA(phi_params=np.zeros(p), theta_params=np.zeros(q))
                
                # Ajustar el modelo y obtener la log-verosimilitud
                fitted_model, log_likelihood = maximum_likelihood_estimation(model_to_fit, 
                                                                                self.data,
                                                                                return_likelihood=True)

                if fitted_model is None:
                    raise ValueError("El ajuste del modelo falló.")

                # Calcular el número de parámetros (p + q + sigma^2)
                # Si se ajustara la constante 'c', sería p + q + 2
                k = p + q + 1 
                
                # Calcular AIC y BIC
                aic = 2 * k - 2 * log_likelihood
                bic = k * np.log(n_obs) - 2 * log_likelihood

                results_list.append({
                    'p': p,
                    'q': q,
                    'log_likelihood': log_likelihood,
                    'aic': aic,
                    'bic': bic,
                    'model': fitted_model
                })
            
            except (np.linalg.LinAlgError, ValueError, IndexError) as e:
                # Si el modelo no converge o falla, lo registramos
                # Usamos tqdm.write para no interferir con la barra de progreso
                tqdm.write(f"  -> Falló el ajuste para ARMA({p},{q}). Error: {e}")
                results_list.append({
                    'p': p, 'q': q, 'log_likelihood': np.nan, 
                    'aic': np.nan, 'bic': np.nan, 'model': None
                })
        
        if not results_list:
            print("La búsqueda no produjo ningún resultado válido.")
            return

        # Convertir resultados a un DataFrame de pandas para fácil manipulación
        self.results_ = pd.DataFrame(results_list).dropna().set_index(['p', 'q'])
        
        if self.results_.empty:
            print("Todos los modelos fallaron en el ajuste.")
            return

        # Encontrar el mejor modelo basado en el criterio especificado
        best_params = self.results_[self.criterion].idxmin()
        self.best_model_ = self.results_.loc[best_params, 'model']
        
        print("\n✅ Grid Search completado.")
        print(f"Mejor modelo encontrado: ARMA({best_params[0]},{best_params[1]}) con {self.criterion.upper()} = {self.results_[self.criterion].min():.2f}")


    def summary(self):
        if self.results_ is None:
            print("El modelo aún no ha sido ajustado. Por favor, ejecuta .fit() primero.")
            return
        
        # Ordenar los resultados por el criterio elegido
        sorted_results = self.results_.sort_values(by=self.criterion)

        print(f"\n--- Resumen de ARMA Grid Search (Ordenado por {self.criterion.upper()}) ---")
        print(sorted_results[['log_likelihood', 'aic', 'bic']].round(2))