import matplotlib.pyplot as plt 
import numpy as np


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"], # La fuente serif por defecto de LaTeX
    "font.sans-serif": ["Computer Modern Sans serif"],
    # Puedes añadir paquetes de LaTeX que necesites para símbolos especiales
    "text.latex.preamble": r"\usepackage{amsmath}" 
})


def plot_irf_comparison(irf_exact, irf_simulation, H):
    periods = np.arange(H)
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].plot(periods, irf_exact, 
            'o-',
            label='Método Exacto (Matricial)', 
            linewidth=2,
            markersize=6,
            color='darkblue')
    ax[0].plot(periods, irf_simulation, 
            'x--',
            label='Método Simulación', 
            color='darkred',
            linewidth=2,
            markersize=7)

    ax[0].set_title('Comparación IRF de un Proceso AR(20)', 
                    fontsize=16)
    ax[0].set_xlabel('Períodos (h)', fontsize=12)
    ax[0].set_ylabel('Respuesta al Impulso', fontsize=12)
    ax[0].axhline(0, color='black', linestyle='--', linewidth=0.8) # Línea en y=0
    ax[0].legend(fontsize=11)
    ax[0].grid(True, which='both', linestyle=':', linewidth=0.7)

    ax[1].plot(periods, irf_exact - irf_simulation, color='k')
    ax[1].set_title('Residuos entre métodos', 
                    fontsize=16)
    ax[1].set_xlabel('Períodos (h)', fontsize=12)
    ax[1].set_ylabel('Diferencia de IRF \n(exacto - simulacion)', fontsize=12)

    plt.savefig("./presentation/figures/irf_comparison.pdf", 
                bbox_inches='tight',
                dpi=300)    
    plt.savefig("./presentation/figures/irf_comparison.png", 
                bbox_inches='tight',
                dpi=300)        
    plt.show()