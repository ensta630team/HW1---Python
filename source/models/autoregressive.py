import numpy as np 
from source.utils.random import generate_stationary_phi


class AR:
    def __init__(self, p=1, params_distribution='stationary'):
        super().__init__()
        self.p = p
        self.phi = None # parameters
        self.F = None # state space matrix
        self.F_modules = None # F eigenvalues absolute value
        self.stationary = None

        if p is not None:
            if params_distribution == 'uniform':
                self.phi = np.random.normal(0, 
                                            0.3, 
                                            size=(1, self.p))
            if params_distribution == 'stationary':
                self.phi = generate_stationary_phi(self.p)
        else:
            self.phi = None

    def set_F(self, F):
        self.F = F

    def set_phi(self, phi):
        if phi is not None: 
            self.p = len(phi)
            self.phi = np.array(phi)

    def build_F(self, phi=None):
        F = np.eye(self.p-1, self.p)
        F = np.vstack([self.phi, F])
        self.set_F(F)
        return F
    
    def is_stationary(self, F=None):
        if F is None: F = self.F
        eigenvalues = np.linalg.eigvals(F)
        modules = abs(eigenvalues)  
        self.stationary = np.all(modules < 1)
        return np.all(modules < 1), modules
    
    def get_irf(self, H=2, method='exact'):
        if self.F is None and method == 'exact':
            raise ValueError("Matrix F was not created. Please check variables p or F.")
        if self.stationary:
            raise ValueError("Process is not statitionary.")
        
        print('⚙️ Using {} method'.format(method))
        irf_values = np.zeros(H) # zero initialized
        irf_values[0] = 1.0 # first IRF always 1

        if method == 'exact':
            for h in range(1, H):
                F_h = np.linalg.matrix_power(self.F, h)
                irf_values[h] = F_h[0, 0]

        if method == 'simulation':
            for h in range(1, H):
                past_values = irf_values[max(0, h - self.p) : h]
                coeffs_to_use = self.phi.flatten()[:len(past_values)]
                irf_values[h] = np.dot(coeffs_to_use, past_values[::-1])
                
        return irf_values