# HW1/source/models/arma.py

import numpy as np
from source.models.base import TimeSeriesModel
from source.models.ar import AutoRegressive
from source.models.ma import MovingAverage
from source.utils.sampling import initialize_params

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
        Calculates the unconditional standard deviation of the ARMA process
        """
        # The exact calculation is complex. For a learning implementation,
        # it's common to omit this or use specialized libraries.
        print("Warning: Unconditional std for ARMA is a complex calculation and is not implemented.")
        return np.nan

    def sample(self, n_samples: int, burn_in: int = 0, **kwargs) -> np.ndarray:
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