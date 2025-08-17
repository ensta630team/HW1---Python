import numpy as np


import numpy as np

class OLS:
    """
    Implements a simple Ordinary Least Squares (OLS) regression model.

    This class allows for fitting a linear model to data and making predictions.
    It automatically handles the inclusion of an intercept term.
    """
    def __init__(self):
        """
        Initializes the OLS model.
        """
        self.beta       = None
        self.intercept_ = None
        self.coef_      = None

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        """
        Adds an intercept (column of ones) to the beginning of the feature matrix X.
        """
        return np.c_[np.ones(X.shape[0]), X]

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fits the OLS model to the training data.

        This method calculates the optimal beta coefficients (including the intercept)
        using the normal equation: β = (X'X)⁻¹X'y.

        Args:
            X (np.ndarray): The training input samples (feature matrix).
                            Shape: (n_samples, n_features).
            y (np.ndarray): The target values (dependent variable).
                            Shape: (n_samples,).
        """
        # Add a column of ones for the intercept term
        X_with_intercept = self._add_intercept(X)

        # Calculate beta coefficients using the normal equation
        factor_0 = np.linalg.inv(np.matmul(X_with_intercept.T, X_with_intercept))
        factor_1 = np.matmul(X_with_intercept.T, y)
        self.beta = np.matmul(factor_0, factor_1)

        # Separate the intercept and the other coefficients for convenience
        self.intercept_ = self.beta[0]
        self.coef_ = self.beta[1:]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Makes predictions using the fitted OLS model.

        Args:
            X (np.ndarray): The input samples for which to make predictions.
                            Shape: (n_samples, n_features).

        Returns:
            np.ndarray: The predicted values. Shape: (n_samples,).
        
        Raises:
            ValueError: If the model has not been fitted yet.
        """
        if self.beta is None:
            raise ValueError("Model has not been fitted yet. Please call the 'fit' method first.")

        # Add intercept to the new data
        X_with_intercept = self._add_intercept(X)

        # Make predictions: y_pred = X * β
        y_pred = np.matmul(X_with_intercept, self.beta)
        return y_pred

