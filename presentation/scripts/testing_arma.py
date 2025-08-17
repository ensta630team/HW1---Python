import numpy as np
from source.models.arma import ARMA, ARMAGridSearch

phi_real = [0.7, 0.2]
theta_real = [-0.4]
true_model = ARMA(c=0.5, sigma=1.0, phi_params=phi_real, theta_params=theta_real)
y_data = true_model.sample(n_samples=200, 
                           burn_in=100)


grid_search = ARMAGridSearch(data_series=y_data, 
                             p_max=10, q_max=10, 
                             criterion='bic')
grid_search.fit()

print(grid_search.summary())
print(grid_search.best_model_.phi, phi_real)
print(grid_search.best_model_.theta, theta_real)