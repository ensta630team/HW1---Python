import matplotlib.pyplot as plt 
import numpy as np
import os
plt.style.use('seaborn-v0_8-whitegrid')

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.sans-serif": ["Computer Modern Sans serif"],
    "text.latex.preamble": r"\usepackage{amsmath}" 
})

FONT_SIZES = {
    'title': 16,
    'label': 14,
    'legend': 12,
    'tick': 12
}

def plot_irf_comparison(irf_exact, irf_simulation, model, ax=None):
    H = len(irf_exact)
    periods = np.arange(H)
    if ax is None:
        fig, ax_array = plt.subplots(1, 2, figsize=(14, 6))
    else:
        ax_array = ax
        fig = ax_array[0].get_figure()

    ax_array[0].plot(periods, irf_exact, 
            'o-',
            label='Método Exacto (Matricial)', 
            linewidth=2,
            markersize=6,
            color='darkblue')
    ax_array[0].plot(periods, irf_simulation, 
            'x--',
            label='Método Simulación', 
            color='darkred',
            linewidth=2,
            markersize=7)

    ax[0].set_title('Comparación IRF de un Proceso \n AR({}) con Horizonte H = {}'.format(model.p, H), 
                    fontsize=FONT_SIZES['title'])
    ax[0].set_xlabel('Períodos (h)', fontsize=FONT_SIZES['label'])
    ax[0].set_ylabel('Respuesta al Impulso', fontsize=FONT_SIZES['label'])
    ax[0].axhline(0, color='black', linestyle='--', linewidth=0.8) # Línea en y=0
    ax[0].grid(True, which='both', linestyle=':', linewidth=0.7)

    ax[1].plot(periods, irf_exact - irf_simulation, color='k')
    ax[1].set_title('Residuos entre métodos', 
                    fontsize=FONT_SIZES['title'])
    ax[1].set_xlabel('Períodos (h)', fontsize=FONT_SIZES['label'])
    ax[1].set_ylabel('Diferencia de IRF \n(exacto - simulacion)', fontsize=FONT_SIZES['label'])
    ax[1].axhline(0, color='black', linestyle='--', linewidth=0.8) # Línea en y=0
    ax[1].grid(True, which='both', linestyle=':', linewidth=0.7)

    # Create a single legend for the figure, placed outside the plots
    handles, labels = ax_array[0].get_legend_handles_labels()
    fig.legend(handles, labels, 
               loc='lower center', 
               bbox_to_anchor=(0.5, -0.14), 
               ncol=2, fontsize=FONT_SIZES['legend'])

    return ax_array

def save_figure(figure, path, **kwargs):
    dirpath = os.path.dirname(path)
    os.makedirs(dirpath, exist_ok=True)
    figure.savefig(path, bbox_inches='tight', dpi=300, **kwargs)  
    print(f"✅ Successfully saved figure to {path}")

def plot_serie(serie, model, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    else:
        fig = ax.get_figure()

    x = np.arange(len(serie))
    ax.plot(x, serie, marker='.', color='darkred')
    ax.set_title('Serie de Tiempo AR({}) {} Periodos'.format(model.p, len(serie)), fontsize=FONT_SIZES['title'])
    ax.set_xlabel('Períodos (h)', fontsize=FONT_SIZES['label'])
    ax.set_ylabel('Valor', fontsize=FONT_SIZES['label'])
    ax.grid(True, which='both', linestyle=':', linewidth=0.7)  

    return ax