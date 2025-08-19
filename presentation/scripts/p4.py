import numpy as np
import matplotlib.pyplot as plt 
import pickle

from source.models.arma import ARMA
from source.utils import plot as uplot 
from source.data.loaders import create_dataset
from source.optimize.conditionalss import conditional_square_sum
from source.optimize.likelihood import maximum_likelihood_estimation
from source.optimize.parallel import parallel_bootstrap_hr
from source.utils.plot import plot_forecast

dataset = create_dataset('./data/hmw1_25.txt', problem=4)

criteria = 'hqic' # hqic bic aic
results_0 = parallel_bootstrap_hr(dataset[:, 0], 
                                  iterations=500, 
                                  j_max=20, 
                                  sampling='parametric',
                                  criteria=criteria,
                                  ols_on=True,
                                  n_jobs=8)

results_1 = parallel_bootstrap_hr(dataset[:, 1], 
                                    iterations=500, 
                                    j_max=20, 
                                    sampling='parametric',
                                    criteria=criteria,
                                    ols_on=True,
                                    n_jobs=8)

results_2 = parallel_bootstrap_hr(dataset[:, 2], 
                                    iterations=500, 
                                    j_max=20, 
                                    sampling='parametric',
                                    criteria=criteria,
                                    ols_on=True,
                                    n_jobs=8)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig, axes = uplot.plot_bootstrap_kde(results_0, fig=fig, axes=axes)
uplot.save_figure(fig, './presentation/figures/p4/pq_{}_0.pdf'.format(criteria))

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig, axes = uplot.plot_bootstrap_kde(results_1, fig=fig, axes=axes)
uplot.save_figure(fig, './presentation/figures/p4/pq_{}_1.pdf'.format(criteria))

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig, axes = uplot.plot_bootstrap_kde(results_2, fig=fig, axes=axes)
uplot.save_figure(fig, './presentation/figures/p4/pq_{}_2.pdf'.format(criteria))

with open('backup.pkl', 'wb') as f:
    pickle.dump([results_0, results_1, results_2], f)