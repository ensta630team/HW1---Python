import numpy as np 


class BaseTS:
    # Base Time Serie Model
    def __init__(self, p=None, q=None):
        # params is a dictonary with relevant parameters
        self.p = p
        self.q = q

    def fit(self):
        raise NotImplementedError("Each model should fit its own method")

    def predict(self, start, end):
        raise NotImplementedError("Each model should fit its own method")

    def save(target):
        pass

class AR(BaseTS):
    def __init__(self, p):
        super().__init__(p)
        if p is None:
            self.phi = np.random.uniform(low=-1, high=1, size=(1, p))
        else:
            self.phi = None

    def build_F(self, phi=None):
        if phi is not None: 
            self.p = len(phi)
            self.phi = phi

        F = np.eye(self.p-1, self.p)
        F = np.vstack([self.phi, F])        
        return F