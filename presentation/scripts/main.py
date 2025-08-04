from source.models.timetools import AR
import numpy as np

# Model instance
ar_model = AR(p=1)

# Create F matrix
F = ar_model.build_F(np.ones(10))
print(F)

# Create F matrix (scalar testing)
F = ar_model.build_F(np.ones(1))
print(F)