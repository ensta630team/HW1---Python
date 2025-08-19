import numpy as np

class Inferencia_OLS:
    def __init__(self, muestra=None, rezagos=None, **kwargs):
        super().__init__()
        self.n             = muestra.shape[0]
        self.muestra       = muestra
        self.rezagos       = rezagos
        self.Y             = self.get_Y()
        self.X             = self.get_X()
        self.beta          = self.get_beta_estimado()
        self.residuos      = self.get_residuos()
        self.var_residuos  = self.get_var_residuos()
        self.momento1      = self.get_momento1()
        self.momento2      = self.get_momento2()

    # self.muestra = y0,y1,y2,y3,y4
    # Y = y2,y3,y4 , equivale a muetsra [2:5]
    # 1 1 1
    # i in range(5): para i 0,1
    # i = 0 muestra[1:4]
    # i = 1 muetsra[0:3]
    
    def get_Y(self):
    # Calculamos los parámetros. En este caso como el AR(2) tiene errores normales, es igual a los estimadores MCO
    # Para esto, tenemos que juntar las matrices
    # Guardamos la variable dependiente
        self.Y = self.muestra[self.rezagos:self.n]
        return self.Y

    def get_X(self):
    # Guardamos las variables independientes
        self.X = np.ones(self.n-self.rezagos)
        for i in range(self.rezagos):
            nuevorezago = self.muestra[self.rezagos-1-i:self.n-1-i]
            #print(nuevorezago.shape, self.X.shape)
            self.X = np.column_stack((self.X, nuevorezago))
        return self.X
    
    def get_beta_estimado(self):
        f0 = np.linalg.inv(np.matmul(self.X.T,self.X))
        f1 = np.matmul(self.X.T,self.Y)
        self.beta = np.matmul(f0,f1)
        return self.beta
    
    def get_residuos(self):
        self.residuos = self.Y - np.matmul(self.X, self.beta)
        return self.residuos
    
    def get_var_residuos(self):
        self.var_residuos = np.matmul(self.residuos.T, self.residuos) / (self.n - 1 - self.rezagos)
        return self.var_residuos
    
    def get_momento1(self):
        # Primer momento de la serie
        self.momento1 = np.mean(self.muestra)
        return self.momento1

    def get_momento2(self):
        # Segundo momento de la serie
        self.momento2 = np.var(self.muestra)
        return self.momento2
    
    def covarianzas(self, orden: int):
        covarianzas = []
        for i in range(orden):
            f1 = self.muestra[i:self.n]
            f1 = f1 - np.mean(f1) * np.ones(self.n-i)
            f2 = self.muestra[0:self.n-i]
            f2 = f2 - np.mean(f2) * np.ones(self.n-i)
            cov = (1/(self.n-i)) * np.sum(np.dot(f1,f2))
            covarianzas.append(cov)
        return covarianzas
    
    def hipotesis(self, R, nula):
        Rb = np.matmul(R,self.beta)
        XX_inv = np.linalg.inv(np.matmul(self.X.T,self.X))
        # Calculamos el estadístico F
        fobs = ((Rb - nula).T @ np.linalg.inv(self.var_residuos * R @ XX_inv @ R.T) @ (Rb - nula)) / 2
        return fobs
    
    def chow(self, quiebre):
        k = self.rezagos +1
        # Resiudos Original
        originales = self.residuos**2
        SC = np.sum(originales)
        # Resiudos Pre-Quiebre
        X_pre = self.X[0:quiebre]
        Y_pre = self.Y[0:quiebre]
        f0 = np.linalg.inv(np.matmul(X_pre.T,X_pre))
        f1 = np.matmul(X_pre.T,Y_pre)
        beta_pre = np.matmul(f0,f1)
        residos_pre = Y_pre - np.matmul(X_pre, beta_pre)
        residos_pre2 = residos_pre**2
        S1 = np.sum(residos_pre2)
        # Resiudos Post-Quiebre
        X_post = self.X[quiebre:]
        Y_post = self.Y[quiebre:]
        f0 = np.linalg.inv(np.matmul(X_post.T,X_post))
        f1 = np.matmul(X_post.T,Y_post)
        beta_post = np.matmul(f0,f1)
        residos_post = Y_post - np.matmul(X_post, beta_post)
        residos_post2 = residos_post**2
        S2 = np.sum(residos_post2)
        # Estadistico Chow
        num = (SC-(S1+S2))/k
        dem = (S1+S2)/(self.X.shape[0]-(2*k))
        chow = num/dem
        return chow