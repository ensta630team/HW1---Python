import numpy as np
from source.utils.sampling import initialize_params
from source.utils.stats import calculate_irf
from scipy.linalg import solve_discrete_lyapunov
from source.models.base import TimeSeriesModel


class AutoRegressive(TimeSeriesModel):
    def __init__(self, 
                 c: float = 0.0, 
                 sigma: float = 1.0,
                 params_distribution=None,
                 **kwargs):
        
        # ==== Class setup ====
        # PHI vector of parameters
        self.phi = initialize_params(params_distribution, **kwargs)
        self.p = len(self.phi) if self.phi is not None else 0

        super().__init__(c, sigma)

    def _is_stationary(self) -> bool:
        """
        Checks for stationarity by inspecting the eigenvalues of the companion matrix F.
        A process is stationary if all eigenvalue moduli are less than 1.
        """
        if self._is_stationary_cached is None:
            self._is_stationary_cached = self._check_stationarity()
        return self._is_stationary_cached


    def get_unconditional_mean(self) -> float:
        """
        Calculates the unconditional mean of the AR process.
        The mean is only defined for a stationary process.
        """

        if self.phi is None:
            return None
        
        if not self._is_stationary:
            return None
        
        phi_sum = np.sum(self.phi)

        if np.isclose(phi_sum, 1.0):
            # Unit root case, mean is not well-defined/infinite
            return np.inf
        
        
        self._unconditional_mean = self.c / (1 - phi_sum)
        return self._unconditional_mean

    def get_unconditional_std(self) -> float:
        """
        Calculates the unconditional standard deviation of the process by solving
        the Yule-Walker equations to find the variance (gamma_0).
        VERSIÓN CORREGIDA: Maneja numéricamente varianzas negativas.
        """
        if not self._is_stationary():
            return np.inf
        
        if self._unconditional_std is None:
            # Cache the result to avoid re-calculating
            gamma_0 = self._solve_yule_walker_for_variance()
            if gamma_0 < 0:
                self._unconditional_std = 0.0
            else:
                self._unconditional_std = np.sqrt(gamma_0)
            # ==========================================
        
        return self._unconditional_std

    def get_irf(self, H: int = 20, method='exact') -> np.ndarray:
        """Calculates the Impulse Response Function (IRF) for H periods."""
        if not self._is_stationary:
            raise ValueError("Process is not stationary. IRF would not converge.")
        if self._is_stationary:
            F = self._build_F_matrix()
        print(f'[INFO] ⚙️ Using {method} method')
        irf_values = calculate_irf(F, H, self.phi, method=method)
        return irf_values

    def sample(self, n_samples: int, initial_values: np.ndarray = None, burn_in: int = 100) -> np.ndarray:
        '''
        The process is defined as:
        y_t = c + phi_1*y_{t-1} + ... + phi_p*y_{t-p} + u_t
        where u_t is a white noise process with standard deviation sigma.

        Args:
            n_samples (int): The number of samples to generate for the final series.
            initial_values (np.ndarray, optional): A 1D array of p starting values. If None,
            initial_values (np.ndarray, optional): A 1D array of p starting values. If None,
                                                   values are drawn from the unconditional distribution
                                                   if stationary, or zeros otherwise.
            burn_in (int): Number of initial samples to generate and discard.
            burn_in (int): Number of initial samples to generate and discard.

        Returns:
        
        '''
        total_samples = n_samples + burn_in
        y_sample = np.zeros(total_samples)
        wnoise = np.random.normal(0, self.sigma, total_samples)
        # Generate the time series
        # Set initial values
        if initial_values is not None:
            y_sample[:self.p] = initial_values
        elif self._is_stationary:
            # Draw from the unconditional distribution if stationary
            y_sample[:self.p] = np.random.normal(self._unconditional_mean, 
                                                 self._unconditional_std, self.p)
        # Otherwise, initial values remain zeros

        # Generate the time series
        for t in range(self.p, total_samples):
            y_prev = y_sample[t-self.p:t]
            # Note: y_prev is already in order [y_{t-p}, ..., y_{t-1}], so we reverse it for dot product
            y_sample[t] = self.c + np.dot(self.phi, y_prev[::-1]) + wnoise[t]
        # Discard the burn-in period
        return y_sample[burn_in:]
    
    # ============================================
    # Private methods ============================
    # ============================================
    def _build_F_matrix(self) -> np.ndarray:
        """
        Constructs the companion matrix F for the AR(p) process
        """
        if self.p == 0:
            return np.empty((0, 0))
            
        F = np.zeros((self.p, self.p))
        # The first row contains the phi coefficients
        F[0, :] = self.phi
        # The subdiagonal contains ones to shift the time series values
        if self.p > 1:
            np.fill_diagonal(F[1:], 1)
        
        return F
    
    def _solve_yule_walker_for_variance(self) -> float:
        """
        This is done by solving the discrete Lyapunov equation for the covariance matrix.
        """
        if not self._is_stationary:
            return np.inf

        if self.p == 0:
            return self.sigma**2
        # We solve the discrete Lyapunov equation: Gamma = F * Gamma * F' + V
        # where Gamma is the unconditional covariance matrix of the state vector,
        # F is the companion matrix, and V is the covariance matrix of the shocks.
        F = self._build_F_matrix()
        V = np.zeros((self.p, self.p))
        V[0, 0] = self.sigma**2

        # solve_discrete_lyapunov solves A*X*A.T - X + Q = 0 for X.
        # We set A = F and Q = -V to match our equation.
        gamma_matrix = solve_discrete_lyapunov(F, V)
        # The unconditional variance of the process (gamma_0) is the top-left element
        gamma_0 = gamma_matrix[0, 0]
        return gamma_0
 
    def _check_stationarity(self) -> bool:
        if self.p == 0:
            return True
            
        F = self._build_F_matrix()
        eigenvalues = np.linalg.eigvals(F)
        return np.all(np.abs(eigenvalues) < 1)