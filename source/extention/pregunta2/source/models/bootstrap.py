import numpy as np
from source.extention.pregunta2.source.utils.bootstrap import remuestracion
from source.extention.pregunta2.source.models.Inferencia import Inferencia_OLS
from source.extention.pregunta2.source.utils.criticos import valores_criticos

class Bootstrap:
    def __init__(self, muestra=None, rezagos=None, n=None, **kwargs):
        super().__init__()
        self.n             = n
        self.muestra       = muestra
        self.rezagos       = rezagos
        self.baseols       = self.get_OLS()
        self.remuestreo    = self.get_remuestra()
        self.bootinferencias = self.get_inferencias()
        self.beta          = self.get_betas()

    def get_remuestra(self):
        remuestreo = []
        for i in range(self.n):
            remuestra_actual = remuestracion(self.muestra, rezagos=self.rezagos)
            remuestreo.append(remuestra_actual)
        self.remuestreo = remuestreo
        return self.remuestreo

    def get_inferencias(self):
        bootinferencias: list[Inferencia_OLS] = []
        for i in range(self.n):
            muestra_actual = self.remuestreo[i]
            # estimacion
            bootinfer = Inferencia_OLS(muestra=muestra_actual, rezagos=self.rezagos)
            # la guardo
            bootinferencias.append(bootinfer)
        self.bootinferencias = bootinferencias    
        return self.bootinferencias

    def get_OLS(self):
        self.baseols = Inferencia_OLS(muestra=self.muestra, rezagos=self.rezagos)
        return self.baseols

    def get_betas(self):
        # calculo phi estimado para cada remuestra
        beta = np.array([]).reshape(0,3)
        # extarigo los phi1 estimados
        for i in range(self.n):
            beta_actual = self.bootinferencias[i].beta
            beta = np.vstack([beta, [beta_actual]])
        self.beta = beta
        return self.beta

    def Efron(self, R):
        phi1s = self.beta[:,R]
        Efron10 = [np.percentile(phi1s, 5), np.percentile(phi1s, 95)]
        Efron05 = [np.percentile(phi1s, 2.5), np.percentile(phi1s, 97.5)]
        Efron01 = [np.percentile(phi1s, 0.5), np.percentile(phi1s, 99.5)]
        print(Efron10, Efron05, Efron01)

    def Hall(self, R):
        phi1s = self.beta[:,R]
        phi2s = np.sort(phi1s) - self.baseols.beta[R]
        Hall10 = [self.baseols.beta[R] - np.percentile(phi2s, 95), self.baseols.beta[R] - np.percentile(phi2s, 5)]
        Hall05 = [self.baseols.beta[R] - np.percentile(phi2s, 97.5), self.baseols.beta[R] - np.percentile(phi2s, 2.5)]
        Hall01 = [self.baseols.beta[R] - np.percentile(phi2s, 99.5), self.baseols.beta[R] - np.percentile(phi2s, 0.5)]
        print(Hall10, Hall05, Hall01)

    def criticos(self, Rh, H0):
        testF = []
        for i in range(self.n):
            fobs = self.bootinferencias[i].hipotesis(R = Rh, nula = H0)
            testF.append(fobs)
        testF = np.array(testF)
        valores_criticos(testF)