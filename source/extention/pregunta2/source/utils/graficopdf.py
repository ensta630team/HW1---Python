import numpy as np
import matplotlib.pyplot as plt 
import scipy.stats as stats
import statsmodels.distributions.empirical_distribution as ECDF
plt.rcParams['text.usetex'] = False  # forzar MathText

from source.extention.pregunta2.source.models.autoregressive import AR
from source.extention.pregunta2.source.utils import plot as uplot
from source.extention.pregunta2.source.models.Inferencia import Inferencia_OLS
from source.extention.pregunta2.source.models.bootstrap import Bootstrap
from source.extention.pregunta2.source.utils.criticos import valores_criticos

def graficopdf(densidad: list): 

    # Parametros de la Distribución Empírica Acumulada ECDF
    x = np.sort(densidad)
    n = x.size
    y = np.arange(1, n+1) / n

    # Parámetros de la Distribución F Acumulada
    dfn = 2   # Grados de libertad del numerador
    dfd = 97  # Grados de libertad del denominador
    q = np.linspace(0, np.max(densidad), 100)

    # Creo la figura y los dos subgráficos
    fig, (ax1, ax2) = plt.subplots(
        nrows=1, ncols=2, # Una fila, dos columnas
        figsize=(12, 6)   # Ajusta el tamaño si es necesario
    )

    # --- Subgráfico 1: ECDF y CDF teórica ---
    # AX1 para el gráfico 1
    ax1.step(x, y, where='post', label='ECDF', color='blue')
    ax1.plot(q, stats.f.cdf(q, dfn, dfd), label='CDF F Teórica', color='green')
    ax1.set_xlabel('Test F')
    ax1.set_ylabel('Probabilidad acumulada')
    ax1.set_title('ECDF empírica vs. CDF teórica del test F')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)  # Para mostrar la grilla

    # --- Subgráfico 2: Densidad empírica y PDF teórica ---
    # AX2 para el segundo gráfico
    ax2.hist(densidad, bins=50, density=True, alpha=0.5, color='blue', label='Densidad empírica')
    ax2.plot(q, stats.f.pdf(q, dfn, dfd), label='PDF F Teórica', color='green')
    ax2.set_xlabel('Test F')
    ax2.set_ylabel('Densidad')
    ax2.set_title('Densidad empírica vs. PDF teórica del test F')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6) # Para mostrar la grilla

    # Ajustar layout y mostrar
    fig.tight_layout()
    #plt.show()