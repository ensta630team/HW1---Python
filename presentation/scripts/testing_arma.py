from source.models.arma import ARMA
from source.optimize.hrissanen import boostrap_hannan_rissanen

from source.utils.plot import plot_hannan_rissanen_results
import matplotlib.pyplot as plt 

phi_real   = [0.7, 0.2]
theta_real = [-0.4]
model = ARMA(c=0.5, sigma=1.0, phi_params=phi_real, theta_params=theta_real)
y = model.sample(n_samples=100, burn_in=100)

results = boostrap_hannan_rissanen(y, 
                                   iterations=300, 
                                   j_max=10,
                                   model=model,
                                   criteria='hqic')


fig, axes = plot_hannan_rissanen_results(results=results)
plt.show()