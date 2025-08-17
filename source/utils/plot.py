import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import seaborn as sns
import os

from source.utils.stats import calculate_acf, calculate_pacf

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


def plot_series_p3(dataset, fig=None, axes=None):
    if fig is None or axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(10, 4))

    x_labels_tick = dataset['t'][1::50]

    axes[0].plot(dataset['t'][1:], dataset['pi_t'], linestyle='-', color='darkred', label=r'$\pi_t$ $(\Delta$ IPC)')
    axes[1].plot(dataset['t'][1:], dataset['y_t'], linestyle='-', color='darkblue', label=r'$y_t$ ($\Delta$ IMACEC)')
    axes[1].set_ylabel('Valor')
    axes[2].plot(dataset['t'], dataset['i_t'], linestyle='-', color='darkgreen', label=r'$i_t$ (PM)')
    axes[2].set_xlabel('Periodo')
    

    for ax in axes:
        ax.set_xticks(x_labels_tick)
        ax.set_xticklabels(x_labels_tick, rotation=0)
        ax.grid(True, linestyle='--', alpha=0.6) # Añadí una grilla para mejor visualización
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=3, frameon=False)
    return fig, axes


def plot_acf_pacf(data: np.ndarray, lags: int = 40, 
                  alpha: float = 0.05, 
                  title_suffix: str = "",
                  fig=None, axes=None):
    """
    Grafica la ACF y PACF de una serie de tiempo.

    Args:
        data (np.ndarray): La serie de tiempo (1D array).
        lags (int): El número máximo de rezagos para graficar.
        alpha (float): Nivel de significancia para los intervalos de confianza (ej. 0.05 para 95% CI).
        title_suffix (str): Sufijo para añadir al título de los gráficos.
    """
    if len(data) < 2:
        print("No hay suficientes datos para calcular ACF/PACF.")
        return

    acf_vals = calculate_acf(data, lags)
    pacf_vals = calculate_pacf(data, lags)

    conf_level = 1.96 / np.sqrt(len(data)) 

    if fig is None or axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Plot ACF
    axes[0].stem(range(lags + 1), acf_vals, markerfmt='k.')
    axes[0].axhspan(-conf_level, conf_level, color='gray', alpha=0.2)
    axes[0].axhline(0, color='k', linestyle='--')
    axes[0].set_title(f'Función de Autocorrelación (ACF) {title_suffix}')
    axes[0].set_ylabel('Autocorrelación')
    axes[0].set_ylim(-1.1, 1.1)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    # Ajuste para el lag 0 en la gráfica
    axes[0].set_xticks(range(0, lags + 1, max(1, lags // 10))) # Ajuste de ticks para no saturar
    lags_for_plot = np.arange(1, lags + 1)
    pacf_vals_for_plot = pacf_vals[1:]
    
    axes[1].stem(lags_for_plot, pacf_vals_for_plot, markerfmt='k.')
    axes[1].axhspan(-conf_level, conf_level, color='gray', alpha=0.2)
    axes[1].axhline(0, color='k', linestyle='--')
    axes[1].set_title(f'Función de Autocorrelación Parcial (PACF) {title_suffix}')
    axes[1].set_xlabel('Rezagos')
    axes[1].set_ylabel('Autocorrelación Parcial')
    axes[1].set_ylim(-1.1, 1.1)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    axes[1].set_xticks(range(0, lags + 1, max(1, lags // 10))) # Ajuste de ticks
    if lags > 0 and lags_for_plot[0] == 0: # Ensure x-axis starts from 1 if lag 0 is nan
        axes[1].set_xlim(0.5, lags + 0.5)

    return fig, axes

def plot_monthly_boxplot(t, y, 
                         title: str = "Distribución Mensual de la Serie", 
                         xlabel: str = "Mes", 
                         ylabel: str = "Valor",
                         fig=None, ax=None):
    """
    Crea un gráfico de cajas (box plot) para visualizar la distribución 
    de una serie de tiempo para cada mes del año.

    Esto es útil para detectar patrones estacionales.

    Args:
        data (pd.Series): La serie de tiempo a analizar. Debe tener un 
                          índice de tipo DatetimeIndex.
        title (str): El título del gráfico.
        xlabel (str): La etiqueta del eje X.
        ylabel (str): La etiqueta del eje Y.
    """
    indice_ym = pd.to_datetime(t, format='%Y/%m')
    data = pd.Series(y, index=indice_ym)


    # 1. Validación de la entrada
    if not isinstance(data, pd.Series):
        raise TypeError("La entrada 'data' debe ser una serie de pandas (pd.Series).")
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("El índice de la serie debe ser de tipo DatetimeIndex.")

    # 2. Preparar los datos para el gráfico
    df = pd.DataFrame({
        'value': data.values,
        'month': data.index.month,
    })

    df = df.sort_values('month')

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(12, 7))

    sns.boxplot(x='month', y='value', 
                data=df, ax=ax, 
                color='#F0F0F0',
                medianprops={'color': '#006400', 'linewidth': 1.5})

    month_names = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
                   'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    ax.set_xticklabels(month_names)

    # 5. Personalizar el resto del gráfico
    ax.set_title(title, fontsize=16, weight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    global_median = df['value'].median()
    ax.axhline(global_median, color='#004080', linestyle='--', linewidth=1., label=f'Mediana Global ({global_median:.3f})')
    ax.legend()

    return fig, ax