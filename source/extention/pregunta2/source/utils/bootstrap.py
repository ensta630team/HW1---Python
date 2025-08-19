import numpy as np
import random
from source.extention.pregunta2.source.models.Inferencia import Inferencia_OLS

def remuestracion(muestra: list, rezagos = int):
    muestra = muestra
    OLS = Inferencia_OLS(muestra=muestra, rezagos=rezagos)
    # Saco residuos
    residuo_base = OLS.residuos
    # Mantiene las primeras porque fueron incondicionales y no hay y-1 para generar despues
    remuestreo = muestra[0:rezagos]
    # Empieza a remuestratr
    for i in range(muestra.shape[0]-rezagos):
        # Toma aleatoriamente uno de los errores
        realizacion = random.randint(0, muestra.shape[0]-1-rezagos)
        residuo_nuevo = residuo_base[realizacion]
        #print(residuo_nuevo)
        # Mantiene la estrcutura de series de tiempo. Es decir en esta nueva muestrar mantenemos la
        # autorgeresividad, por lo que, por ejemplo, y10 se explica con y9 y y8 remuestrados.
        predictor = np.append(remuestreo[i:i+rezagos],1)
        # Ahora vuelvo a tener yt usando el predictor anterior mas el error remuestra
        observacion = np.sum(np.dot(OLS.beta[::-1],predictor)) + residuo_nuevo
        #print(observacion)
        remuestreo = np.append(remuestreo, observacion)
        #print(remuestreo)
    return remuestreo
