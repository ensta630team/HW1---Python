import numpy as np
from abc import ABC, abstractmethod
from source.utils.print import check_num


class TimeSeriesModel(ABC):
    """
    Abstract base class for univariate time series models.

    This class provides a common interface for AR, MA, and ARMA models,
    defining essential properties and methods that all subclasses must implement.
    It serves as a "contract" to ensure consistency across different model types.
    """

    def __init__(self, 
                 c: float = 0.0, 
                 sigma: float = 1.0):
        """
        Initializes the base time series model.

        Args:
            c (float): The constant term of the model. For a stationary model,
                       this value is directly related to the unconditional mean.
            sigma (float): The standard deviation of the white noise error term (u_t).
                           Must be a positive value.
        """
        if sigma <= 0:
            raise ValueError("Sigma (standard deviation of the error term) must be positive.")
        
        self.c      = c # model's constant
        self.sigma  = sigma # error's deviation

        # This will call the child class's specific implementation of the method.
        self._is_stationary_cached = None
        self._unconditional_mean   = None
        self._unconditional_std    = None

        self._unconditional_mean = self.get_unconditional_mean()
        self._unconditional_std  = self.get_unconditional_std()

    @abstractmethod
    def get_unconditional_mean(self) -> float:
        """
        Calculates the unconditional mean (expected value) of the process.
        This method MUST be implemented by each subclass.
        """
        pass

    @abstractmethod
    def get_unconditional_std(self) -> float:
        """
        Calculates the unconditional variance of the process.
        This method MUST be implemented by each subclass.
        """
        pass

    @abstractmethod
    def get_irf(self, H: int = 20) -> np.ndarray:
        """
        Calculates the Impulse Response Function (IRF) for the model.
        This method MUST be implemented by each subclass.

        Args:
            H (int): The number of periods (horizon) for which to calculate the IRF.

        Returns:
            np.ndarray: An array of length H containing the IRF values.
        """
        pass

    @abstractmethod
    def sample(self, n_samples: int, initial_values: np.ndarray = None, burn_in: int = 100) -> np.ndarray:
        """
        Generates a time series sample from the model.
        This method MUST be implemented by each subclass.

        Args:
            n_samples (int): The desired length of the final time series.
            burn_in (int): The number of initial samples to generate and discard
                           to allow the process to stabilize.

        Returns:
            np.ndarray: The generated time series of length `n_samples`.
        """
        pass

    @abstractmethod
    def _is_stationary(self) -> bool:
        """
        Checks if the model is stationary.
        This method MUST be implemented by each subclass.

        Returns:
            bool: True if the model is stationary, False otherwise.                     
        """
        pass


    def __str__(self) -> str:
        """Returns a string summary of the process."""
        
        is_stationary = self._is_stationary()

        if 'p' in self.__dict__.keys():
            header = f"AR({self.p}) Model Summary"
            order =  self.__dict__['p']
            coefs  =  self.__dict__['phi']
            coefname = 'phi'
            
        if 'q' in self.__dict__.keys():
            header = f"MA({self.q}) Model Summary"
            order =  self.__dict__['q']
            coefs =  self.__dict__['theta']
            coefname = 'theta'
                    
        separator = "=" * 50

        summary_lines = [
            separator,
            f"{header:^50}",
            separator,
            f"{'Model Order (p)':<25}: {check_num(order)}",
            f"{'Intercept (c)':<25}: {check_num(self.c)}",
            f"{'Error Std Dev (sigma)':<25}: {check_num(self.sigma)}",
            f"{'Is Stationary':<25}: {is_stationary}",
            f"{'Mu (Unc. Mean):':<25}: {check_num(self._unconditional_mean)}",
            f"{'Sigma (Unc. Std.)':<25}: {check_num(self._unconditional_std)}",
        ]
        
        summary_lines.append("-" * 50)
        summary_lines.append(f"Coefficients ({coefname}):")
        
        if order > 0:
            for i, coef in enumerate(coefs):
                summary_lines.append(f"  {coefname}_{i+1:<4} = {coef: >5.2f}")
        else:
            summary_lines.append("  (No coefficients for AR(0) model)")
            
        summary_lines.append(separator)
        
        return "\n".join(summary_lines)