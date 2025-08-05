import numpy as np 
from source.utils.random import generate_stationary_phi

class BaseTS:
    # Base Time Serie Model
    def __init__(self, p=None, q=None):
        # params is a dictonary with relevant parameters
        self.p = p
        self.q = q

class AR(BaseTS):
    def __init__(self, p, dist='stationary'):
        super().__init__(p)
        if p is not None:
            if dist == 'uniform':
                self.phi = np.random.normal(0, 
                                            0.3, 
                                            size=(1, self.p))
            if dist == 'stationary':
                self.phi = generate_stationary_phi(self.p)
        else:
            self.phi = None

    def build_F(self, phi=None):
        if phi is not None: 
            self.p = len(phi)
            self.phi = phi
        F = np.eye(self.p-1, self.p)
        F = np.vstack([self.phi, F])        
        return F
    
    def is_stationary(self, F):
        eigenvalues = np.linalg.eigvals(F)
        modules = abs(eigenvalues)  
        return np.all(modules < 1), modules
    
    def get_irf(self, F, H=2, method='exact'):
        print('[INFO] Using {} method'.format(method))
        irf_values = np.zeros(H)
        irf_values[0] = 1.0
        if method == 'exact':
            for h in range(1, H):
                F_h = np.linalg.matrix_power(F, h)
                irf_values[h] = F_h[0, 0]

        if method == 'simulation':
            for h in range(1, H):
                past_values = irf_values[max(0, h - self.p) : h]
                coeffs_to_use = self.phi.flatten()[:len(past_values)]
                irf_values[h] = np.dot(coeffs_to_use, past_values[::-1])
                
        return irf_values