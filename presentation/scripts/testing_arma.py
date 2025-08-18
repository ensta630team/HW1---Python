import numpy as np
from source.models.arma import ARMA
from source.optimize.hrissanen import HannanRissanen




phi_real   = [0.7, 0.2]
theta_real = [-0.4]


values_pred_ar = []
values_pred_ma = []
for i in range(10):
    true_model = ARMA(c=0.5, sigma=1.0, phi_params=phi_real, theta_params=theta_real)
    y_data = true_model.sample(n_samples=1000, 
                            burn_in=100)
    best_p, best_q, final_model, results_df = HannanRissanen(y_data, j_max=4, criteria='hqic')
    values_pred_ar.append(final_model.phi)
    values_pred_ma.append(final_model.theta)




