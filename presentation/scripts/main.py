import numpy as np

from source.models.timetools import AR
from source.utils.plot import plot_irf_comparison

# Model instance
H = 20
p=20
ar_model = AR(p=p)

F = ar_model.build_F()

cond_0 = ar_model.is_stationary(F)

irf_exact = ar_model.get_irf(H=H, F=F, method='exact')
irf_simulation = ar_model.get_irf(H=H, F=F, 
                                  method='simulation')

plot_irf_comparison(irf_exact, irf_simulation, H)
