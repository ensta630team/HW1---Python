import numpy as np 
from source.utils.sampling import initialize_params
from source.models.base import TimeSeriesModel


class MovingAverage(TimeSeriesModel):
    """
    Implements a Moving Average (MA) model.

    This class defines an MA(q) process, provides methods for analyzing its
    properties like stationarity and invertibility, and allows for the generation
    of sample data.

    The process is defined as:
    y_t = c + u_t + θ₁*u_{t-1} + ... + θq*u_{t-q}
    """
    def __init__(self,
                 c: float = 0.0,
                 sigma: float = 1.0,
                 params_distribution=None,
                 **kwargs):
        """
        Initializes the MA model.

        Args:
            c (float): The constant term of the model, which is equal to the
                       unconditional mean (mu).
            sigma (float): The standard deviation of the white noise error term.
            params_distribution (list or np.ndarray, optional): A vector of the
                                 MA parameters (theta coefficients).
            **kwargs: Additional arguments passed to the parameter initializer.
        """
        self.theta = initialize_params(params_distribution, **kwargs)
        self.q = len(self.theta) if self.theta is not None else 0

        super().__init__(c, sigma)

    def _is_stationary(self) -> bool:
        """
        Checks for stationarity. All finite-order MA(q) processes are
        weakly stationary by definition.
        """
        return True

    def get_unconditional_mean(self) -> float:
        """
        Calculates the unconditional mean of the MA process.
        For an MA model, the mean is simply the constant 'c'.
        """
        return self.c

    def get_unconditional_std(self) -> float:
        """
        Calculates the unconditional standard deviation of the MA process.
        The variance is σ² * (1 + θ₁² + θ₂² + ... + θq²).
        """
        if self.theta is None: return None
        theta_sq_sum = np.sum(self.theta**2)
        variance = self.sigma**2 * (1 + theta_sq_sum)
        return np.sqrt(variance)

    def is_invertible(self) -> bool:
        """
        Checks for invertibility by finding the roots of the MA characteristic polynomial.
        The process is invertible if all roots lie outside the unit circle.

        Returns:
            bool: True if the model is invertible, False otherwise.
        """
        if self.q == 0:
            return True

        # Polynomial coefficients: 1, theta_1, theta_2, ..., theta_q
        coeffs = np.concatenate(([1], self.theta))
        roots = np.roots(coeffs)

        # Invertibility requires the modulus of all roots to be > 1
        return np.all(np.abs(roots) > 1)

    def get_irf(self, H: int = 20) -> np.ndarray:
        """
        Calculates the Impulse Response Function (IRF) for an MA(q) process.
        The IRF is finite and corresponds to [1, θ₁, ..., θq, 0, ...].

        Args:
            H (int): The number of periods (horizon) for which to calculate the IRF.

        Returns:
            np.ndarray: An array of length H containing the IRF values.
        """
        irf_values = np.zeros(H)
        irf_values[0] = 1  # The impact at t=0 is 1

        # The number of coefficients to use is min(H-1, q)
        num_coeffs_to_use = min(H - 1, self.q)
        if num_coeffs_to_use > 0:
            irf_values[1:num_coeffs_to_use + 1] = self.theta[:num_coeffs_to_use]

        return irf_values

    def sample(self, n_samples: int, burn_in: int = 100, **kwargs) -> np.ndarray:
        """
        Generates a time series sample from the MA(q) process by manually
        iterating through time.

        The process is defined as:
        y_t = c + u_t + θ₁*u_{t-1} + ... + θq*u_{t-q}

        Args:
            n_samples (int): The desired length of the final time series.
            burn_in (int): The number of initial samples to generate and discard
                           to allow the process to stabilize.

        Returns:
            np.ndarray: The generated time series of length `n_samples`.
        """
        total_samples = n_samples + burn_in

        # 1. Generate all white noise shocks at once
        shocks = np.random.normal(loc=0, scale=self.sigma, size=total_samples)

        # 2. Initialize the array for the time series
        y_sample = np.zeros(total_samples)

        # 3. Iterate through time to build the series
        # The loop starts at 'q' because we need 'q' past shocks for the first calculation
        for t in range(self.q, total_samples):
            # Get the last q shocks (from u_{t-1} down to u_{t-q})
            past_shocks = shocks[t-self.q:t]

            # Calculate the weighted sum of past shocks
            # The past_shocks array is reversed (`[::-1]`) to align with the theta order (θ₁, θ₂, ...)
            ma_component = np.dot(self.theta, past_shocks[::-1])

            # Calculate y_t using the MA(q) formula
            y_sample[t] = self.c + shocks[t] + ma_component

        # 4. Discard the burn-in period
        return y_sample[burn_in:]