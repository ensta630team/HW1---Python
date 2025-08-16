import numpy as np 
from source.utils.random import initialize_params
from source.utils.ts import calculate_irf


class AR:
    def __init__(self, params_distribution=None, c=0.0, sigma=1.0, **kwargs):
        super().__init__()
        self.c          = c # Intercept
        self.sigma      = sigma # Standard deviation of the error term
        self.F          = None # state space matrix
        self.F_modules  = None # F eigenvalues absolute value
        self.stationary = None # boolean if satisfies     
        self.phi        = initialize_params(params_distribution, **kwargs)
        self.p          = len(self.phi) if self.phi is not None else 0
        self.mu         = self.get_unconditional_mean()
        self.ustd       = self.get_unconditional_std()

    def get_unconditional_mean(self):
        """
        Calculates the unconditional mean of the AR process.
        The mean is only defined for a stationary process.
        """
        if self.stationary is None:
            self.is_stationary()
        
        if not self.stationary:
            return None
            
        return self.c / (1 - np.sum(self.phi))

    def get_unconditional_std(self):
        """
        Calculates the unconditional variance of the AR process.
        The variance is only defined for a stationary process.
        """
        if self.stationary is None:
            self.is_stationary()
        if not self.stationary:
            return None
        
        return np.sqrt(self.sigma**2 / (1 - np.sum(self.phi**2)))
        
    def build_F(self):
        """Builds the companion matrix F from the model's phi coefficients."""
        if self.p == 0:
            self.F = np.array([[]])
        else:
            top_row = self.phi.reshape(1, self.p)
            bottom_part = np.eye(self.p - 1, self.p)
            self.F = np.vstack([top_row, bottom_part])
        return self.F
    
    def is_stationary(self):
        """
        Checks for stationarity by inspecting the eigenvalues of the companion matrix F.
        A process is stationary if all eigenvalue moduli are less than 1.
        """
        if self.F is None:
            self.build_F()
        
        if self.p == 0: # An AR(0) is stationary by definition
            self.stationary = True
            self.F_modules = np.array([])
            return True, self.F_modules

        eigenvalues = np.linalg.eigvals(self.F)
        self.F_modules = np.abs(eigenvalues)
        self.stationary = np.all(self.F_modules < 1)
        return self.stationary, self.F_modules
    
    def get_irf(self, H=2, method='exact'):
        """Calculates the Impulse Response Function (IRF) for H periods."""
        # Ensure model state is calculated before proceeding
        if self.stationary is None:
            self.is_stationary()

        if not self.stationary:
            raise ValueError("Process is not stationary. IRF would not converge.")
        
        print(f'⚙️ Using {method} method')
        irf_values = calculate_irf(self.F, H, self.phi, method=method)
        
        return irf_values
   
    def sample(self, n_samples: int, initial_values: np.ndarray = None):
        """
        The process is defined as:
        y_t = c + phi_1*y_{t-1} + ... + phi_p*y_{t-p} + u_t
        where u_t is a white noise process with standard deviation sigma.

        Args:
            n_samples (int): The number of samples to generate for the final series.
            initial_values (np.ndarray, optional): A 1D array of p starting values.
                                                   If None, starts with zeros. Defaults to None.

        Returns:
            np.ndarray: A 1D numpy array of size n_samples representing the generated time series.
            f"{'Intercept (c)':<25}: {self.c}",
            f"{'Error Std Dev (sigma)':<25}: {self.sigma}",
        """
        if initial_values is not None and len(initial_values) != self.p:
            raise ValueError(f"initial_values must have length p={self.p}, but got {len(initial_values)}.")

        # first 2 observations
        y_init   = np.random.normal(self.mu, self.ustd, self.p)
        
        y_sample = np.zeros([n_samples - self.p]) 
        y_sample = np.concatenate((y_init, y_sample), axis=0)
        wnoise   = np.random.normal(0, self.sigma, n_samples-self.p)
 
        for t in range(self.p, n_samples):
            y_prev = y_sample[t-self.p:t]
            y_sample[t] = self.c + np.dot(self.phi, y_prev[::-1]) + wnoise[t-self.p]
            
        return y_sample

    def __str__(self):
        """Returns a string summary of the AR model properties."""
        if self.stationary is None:
            self.is_stationary()

        header = f"AR({self.p}) Model Summary"
        separator = "=" * 50

        summary_lines = [
            separator,
            f"{header:^50}",
            separator,
            f"{'Model Order (p)':<25}: {self.p}",
            f"{'Intercept (c)':<25}: {self.c}",
            f"{'Error Std Dev (sigma)':<25}: {self.sigma}",
            f"{'Is Stationary':<25}: {self.stationary}",
            f"{'Mu (Unc. Mean):':<25}: {round(self.mu, 2) if self.mu is not None else 'N/A'}",
            f"{'Sigma (Unc. Std.)':<25}: {round(self.ustd, 2) if self.ustd is not None else 'N/A'}",
        ]

        if self.F_modules is not None and len(self.F_modules) > 0:
            max_modulus = np.max(self.F_modules)
            summary_lines.append(f"{'Max Eigenvalue Modulus':<25}: {max_modulus:.4f}")
        
        summary_lines.append("-" * 50)
        summary_lines.append("Coefficients (phi):")
        
        if self.p > 0:
            for i, coef in enumerate(self.phi):
                summary_lines.append(f"  phi_{i+1:<4} = {coef: >10.4f}")
        else:
            summary_lines.append("  (No coefficients for AR(0) model)")
            
        summary_lines.append(separator)
        
        return "\n".join(summary_lines)