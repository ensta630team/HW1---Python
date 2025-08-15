from source.models.autoregressive import AR
import numpy as np

phi_vector = [0.8, -0.2]
model = AR(c=1, sigma=0.2, params_distribution=phi_vector)
model.sample(n_samples=10)
# print(model)

