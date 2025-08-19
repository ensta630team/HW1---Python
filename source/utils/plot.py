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
    """
    Plots a comparison between the exact and simulated Impulse Response Functions (IRF).

    Args:
        irf_exact (np.ndarray): The IRF calculated using the exact matrix method.
        irf_simulation (np.ndarray): The IRF calculated using simulation.
        model: The model object, which should have a 'p' attribute for the AR order.
        ax (matplotlib.axes.Axes, optional): A pre-existing array of two axes to plot on. Defaults to None.

    Returns:
        matplotlib.axes.Axes: The array of axes used for the plot.
    """
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

    ax_array[0].set_title('Comparación IRF de un Proceso \n AR({}) con Horizonte H = {}'.format(model.p, H),
                    fontsize=FONT_SIZES['title'])
    ax_array[0].set_xlabel('Períodos (h)', fontsize=FONT_SIZES['label'])
    ax_array[0].set_ylabel('Respuesta al Impulso', fontsize=FONT_SIZES['label'])
    ax_array[0].axhline(0, color='black', linestyle='--', linewidth=0.8) # Línea en y=0
    ax_array[0].grid(True, which='both', linestyle=':', linewidth=0.7)
    ax_array[0].tick_params(axis='both', which='major', labelsize=FONT_SIZES['tick'])

    ax_array[1].plot(periods, irf_exact - irf_simulation, color='k')
    ax_array[1].set_title('Residuos entre métodos',
                    fontsize=FONT_SIZES['title'])
    ax_array[1].set_xlabel('Períodos (h)', fontsize=FONT_SIZES['label'])
    ax_array[1].set_ylabel('Diferencia de IRF \n(exacto - simulacion)', fontsize=FONT_SIZES['label'])
    ax_array[1].axhline(0, color='black', linestyle='--', linewidth=0.8) # Línea en y=0
    ax_array[1].grid(True, which='both', linestyle=':', linewidth=0.7)
    ax_array[1].tick_params(axis='both', which='major', labelsize=FONT_SIZES['tick'])

    # Create a single legend for the figure, placed outside the plots
    handles, labels = ax_array[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='lower center',
               bbox_to_anchor=(0.5, -0.14),
               ncol=2, fontsize=FONT_SIZES['legend'])

    return ax_array

def save_figure(figure, path, **kwargs):
    """
    Saves a matplotlib figure to a specified path.

    Args:
        figure (matplotlib.figure.Figure): The figure to save.
        path (str): The path where the figure will be saved.
        **kwargs: Additional arguments to pass to `figure.savefig`.
    """
    dirpath = os.path.dirname(path)
    os.makedirs(dirpath, exist_ok=True)
    figure.savefig(path, bbox_inches='tight', dpi=300, **kwargs)
    print(f"✅ Successfully saved figure to {path}")

def plot_serie(serie, model, ax=None, **kwargs):
    """
    Plots a time series.

    Args:
        serie (np.ndarray): The time series data to plot.
        model: The model object, which should have a 'p' attribute for the AR order.
        ax (matplotlib.axes.Axes, optional): A pre-existing axis to plot on. Defaults to None.

    Returns:
        matplotlib.axes.Axes: The axis used for the plot.
    """
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
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZES['tick'])

    return ax


def plot_series_p3(dataset, fig=None, axes=None):
    """
    Plots three time series from a dataset in separate subplots.

    Args:
        dataset (pd.DataFrame): DataFrame containing the time series data with columns 't', 'pi_t', 'y_t', and 'i_t'.
        fig (matplotlib.figure.Figure, optional): A pre-existing figure. Defaults to None.
        axes (matplotlib.axes.Axes, optional): A pre-existing array of three axes. Defaults to None.

    Returns:
        tuple: A tuple containing the figure and the array of axes.
    """
    if fig is None or axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(10, 4))

    x_labels_tick = dataset['t'][1::50]

    axes[0].plot(dataset['t'][1:], dataset['pi_t'], linestyle='-', color='darkred', label=r'$\pi_t$ $(\Delta$ IPC)')
    axes[1].plot(dataset['t'][1:], dataset['y_t'], linestyle='-', color='darkblue', label=r'$y_t$ ($\Delta$ IMACEC)')
    axes[1].set_ylabel('Valor', fontsize=FONT_SIZES['label'])
    axes[2].plot(dataset['t'], dataset['i_t'], linestyle='-', color='darkgreen', label=r'$i_t$ (PM)')
    axes[2].set_xlabel('Periodo', fontsize=FONT_SIZES['label'])


    for ax in axes:
        ax.set_xticks(x_labels_tick)
        ax.set_xticklabels(x_labels_tick, rotation=0, fontsize=FONT_SIZES['tick'])
        ax.grid(True, linestyle='--', alpha=0.6) # Add a grid for better visualization
        ax.tick_params(axis='y', which='major', labelsize=FONT_SIZES['tick'])

    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=3, frameon=False, fontsize=FONT_SIZES['legend'])
    return fig, axes


def plot_acf_pacf(data: np.ndarray, lags: int = 40,
                  alpha: float = 0.05,
                  title_suffix: str = "",
                  fig=None, axes=None):
    """
    Plots the ACF and PACF of a time series.

    Args:
        data (np.ndarray): The time series (1D array).
        lags (int): The maximum number of lags to plot.
        alpha (float): Significance level for confidence intervals (e.g., 0.05 for 95% CI).
        title_suffix (str): Suffix to add to the plot titles.
        fig (matplotlib.figure.Figure, optional): A pre-existing figure. Defaults to None.
        axes (matplotlib.axes.Axes, optional): A pre-existing array of two axes. Defaults to None.

    Returns:
        tuple: A tuple containing the figure and the array of axes.
    """
    if len(data) < 2:
        print("Not enough data to calculate ACF/PACF.")
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
    axes[0].set_title(f'Función de Autocorrelación (ACF) {title_suffix}', fontsize=FONT_SIZES['title'])
    # axes[0].set_xlabel('Rezagos', fontsize=FONT_SIZES['label'])
    axes[0].set_ylabel('Autocorrelación', fontsize=FONT_SIZES['label'])
    maxv =  np.max(acf_vals[acf_vals!=1])
    minv = np.min(acf_vals)
 
    axes[0].set_ylim(minv-(abs(minv)/5), maxv+(abs(maxv)/10))
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].tick_params(axis='both', which='major', labelsize=FONT_SIZES['tick'])
    axes[0].set_xticks(range(0, lags + 1, max(1, lags // 10))) # Adjust ticks to avoid clutter

    # Plot PACF
    lags_for_plot = np.arange(1, lags + 1)
    pacf_vals_for_plot = pacf_vals[1:]

    axes[1].stem(lags_for_plot, pacf_vals_for_plot, markerfmt='k.')
    axes[1].axhspan(-conf_level, conf_level, color='gray', alpha=0.2)
    axes[1].axhline(0, color='k', linestyle='--')
    axes[1].set_title(f'Función de Autocorrelación Parcial (PACF) {title_suffix}', fontsize=FONT_SIZES['title'])
    axes[1].set_xlabel('Rezagos', fontsize=FONT_SIZES['label'])
    axes[1].set_ylabel('Autocorrelación Parcial', fontsize=FONT_SIZES['label'])
    # axes[1].set_ylim(-1.1, 1.1)
    maxv =  np.max(pacf_vals_for_plot)
    minv = np.min(pacf_vals_for_plot)
    axes[1].set_ylim(minv-(abs(minv)/5), maxv+maxv/5)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].tick_params(axis='both', which='major', labelsize=FONT_SIZES['tick'])
    axes[1].set_xticks(range(0, lags + 1, max(1, lags // 10))) # Adjust ticks
    if lags > 0 and lags_for_plot.all() != 0:
        axes[1].set_xlim(0.5, lags + 0.5)

    return fig, axes

def plot_monthly_boxplot(t, y,
                         title: str = "Distribución Mensual de la Serie",
                         xlabel: str = "Mes",
                         ylabel: str = "Valor",
                         fig=None, ax=None):
    """
    Creates a box plot to visualize the distribution of a time series for each month of the year.
    This is useful for detecting seasonal patterns.

    Args:
        t (array-like): Time component, e.g., in 'YYYY/MM' format.
        y (array-like): Value component of the time series.
        title (str): The title of the plot.
        xlabel (str): The label for the X-axis.
        ylabel (str): The label for the Y-axis.
        fig (matplotlib.figure.Figure, optional): A pre-existing figure. Defaults to None.
        ax (matplotlib.axes.Axes, optional): A pre-existing axis. Defaults to None.

    Returns:
        tuple: A tuple containing the figure and the axis.
    """
    indice_ym = pd.to_datetime(t, format='%Y/%m')
    data = pd.Series(y, index=indice_ym)

    # 1. Input validation
    if not isinstance(data, pd.Series):
        raise TypeError("The input 'data' must be a pandas Series (pd.Series).")
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("The Series index must be of type DatetimeIndex.")

    # 2. Prepare data for plotting
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
    ax.set_xticklabels(month_names, fontsize=FONT_SIZES['tick'])

    # 5. Customize the rest of the plot
    ax.set_title(title, fontsize=FONT_SIZES['title'], weight='bold')
    ax.set_xlabel(xlabel, fontsize=FONT_SIZES['label'])
    ax.set_ylabel(ylabel, fontsize=FONT_SIZES['label'])
    ax.tick_params(axis='y', which='major', labelsize=FONT_SIZES['tick'])


    global_median = df['value'].median()
    ax.axhline(global_median, color='#004080', linestyle='--', linewidth=1., label=f'Mediana Global ({global_median:.3f})')
    ax.legend(fontsize=FONT_SIZES['legend'])

    return fig, ax

def plot_hannan_rissanen_results(results, fig=None, axes=None):
    """
    Plots the distribution of the best p and q orders from the Hannan-Rissanen algorithm results.

    Args:
        results (dict): A dictionary containing the results, including 'freq_p', 'freq_q', 'best_p', and 'best_q'.
        fig (matplotlib.figure.Figure, optional): A pre-existing figure. Defaults to None.
        axes (matplotlib.axes.Axes, optional): A pre-existing array of two axes. Defaults to None.
    """
    freq_p = results['freq_p']
    print(freq_p)
    print(results['best_p'])
    freq_q = results['freq_q']
    if fig is None or axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(7, 3))

    # sns.kdeplot(freq_p, ax=axes[0], color='darkblue')
    # sns.kdeplot(freq_q, ax=axes[1], color='darkblue')

    sns.barplot(x=np.arange(0, freq_p.shape[0], 1), 
                y=freq_p, 
                ax=axes[0], color='darkblue', alpha=0.5)
    sns.barplot(x=np.arange(0, freq_q.shape[0], 1), 
                y=freq_q, 
                ax=axes[1], 
                color='darkblue', alpha=0.5)
    
    axes[1].plot(results['best_q'], freq_q[results['best_q']], "*", 
                 markersize=10, color="darkred", 
                 label=f'Mejor q: {results["best_q"]}')
    
    axes[0].plot(results['best_p'], freq_p[results['best_p']], "*", 
                 markersize=10, color="darkred", 
                 label=f'Mejor p: {results["best_p"]}')
    axes[0].set_xticks(np.arange(0, freq_q.shape[0], 2))
    axes[1].set_xticks(np.arange(0, freq_q.shape[0], 2))
    # axes[0].axvline(x=results['best_p'], color='red', linestyle='--', label=f'Mejor p: {results["best_p"]}')
    # axes[1].axvline(x=results['best_q'], color='red', linestyle='--', label=f'Mejor q: {results["best_q"]}')

    # axes[0].set_title('Distribución del Mejor p', fontsize=FONT_SIZES['title'])
    # axes[1].set_title('Distribución del Mejor q', fontsize=FONT_SIZES['title'])

    axes[0].set_xlabel('Valores de p', fontsize=FONT_SIZES['label'])
    axes[1].set_xlabel('Valores de q', fontsize=FONT_SIZES['label'])

    axes[0].set_ylabel('Densidad', fontsize=FONT_SIZES['label'])

    axes[0].legend(fontsize=FONT_SIZES['legend'],
                    facecolor='whitesmoke', 
                    edgecolor='gray',
                    shadow=True)
    axes[1].legend(fontsize=FONT_SIZES['legend'],
                    facecolor='whitesmoke', 
                    edgecolor='gray',
                    shadow=True)


    for ax in axes:
        ax.tick_params(axis='both', which='major', labelsize=FONT_SIZES['tick'])

    return fig, axes


def plot_bootstrap_kde(bootstrap_results: dict, fig=None, axes=None):
    """
    Visualiza la distribución de los órdenes p y q del bootstrap
    usando gráficos de Densidad de Kernel (KDE).

    Args:
        bootstrap_results (dict): El diccionario devuelto por la función
                                  `parametric_bootstrap_hr`.
        fig (matplotlib.figure.Figure, optional): Figura preexistente.
        axes (matplotlib.axes.Axes, optional): Array de dos ejes preexistente.
    """
    # 1. Validación y extracción de datos
    if 'param_distribution' not in bootstrap_results:
        raise ValueError("La entrada debe ser el diccionario de resultados de la función de bootstrap.")

    dist_df = bootstrap_results['param_distribution']
    best_p = bootstrap_results['best_p']
    best_q = bootstrap_results['best_q']
    
    # 2. Reconstruir las listas completas de todos los p y q encontrados
    #    Esto es necesario para que el KDE pueda estimar la densidad.
    all_p = []
    all_q = []
    for _, row in dist_df.iterrows():
        p, q = row['(p,q)']
        frequency = int(row['frequency'])
        all_p.extend([p] * frequency)
        all_q.extend([q] * frequency)

    # 3. Creación de los gráficos
    if fig is None or axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Gráfico para p ---
    sns.kdeplot(x=all_p, ax=axes[0], color='darkblue', linewidth=2.5, fill=True, alpha=0.1)
    axes[0].axvline(x=best_p, color='red', linestyle='--', linewidth=2, label=f'Mejor p: {best_p}')
    axes[0].set_title('Distribución del Mejor p', fontsize=FONT_SIZES['title'])
    axes[0].set_xlabel('Orden AR (p)', fontsize=FONT_SIZES['label'])
    axes[0].set_ylabel('Densidad', fontsize=FONT_SIZES['label'])
    axes[0].legend(fontsize=FONT_SIZES['legend'])
    axes[0].grid(True, which='both', linestyle=':', linewidth=0.7)
    axes[0].tick_params(axis='both', which='major', labelsize=FONT_SIZES['tick'])
    axes[0].set_xlim(left=-0.5) # Asegurar que el eje no empiece en negativo

    # --- Gráfico para q ---
    sns.kdeplot(x=all_q, ax=axes[1], color='darkblue', linewidth=2.5, fill=True, alpha=0.1)
    axes[1].axvline(x=best_q, color='red', linestyle='--', linewidth=2, label=f'Mejor q: {best_q}')
    axes[1].set_title('Distribución del Mejor q', fontsize=FONT_SIZES['title'])
    axes[1].set_xlabel('Orden MA (q)', fontsize=FONT_SIZES['label'])
    axes[1].set_ylabel('Densidad', fontsize=FONT_SIZES['label'])
    axes[1].legend(fontsize=FONT_SIZES['legend'])
    axes[1].grid(True, which='both', linestyle=':', linewidth=0.7)
    axes[1].tick_params(axis='both', which='major', labelsize=FONT_SIZES['tick'])
    axes[1].set_xlim(left=-0.5)

    plt.tight_layout()
    plt.show()
    
    return fig, axes

def plot_forecast(train_t, train_y, test_t, test_y, forecasts, lower_ci, upper_ci, 
                  model_name="ARMA", fig=None, axes=None):
    """
    Crea un gráfico con dos subplots para visualizar los pronósticos y los residuos.
    También calcula y muestra el RMSE y el R-cuadrado.

    Args:
        train_t, train_y: Datos de tiempo y valores de entrenamiento.
        test_t, test_y: Datos de tiempo y valores reales de prueba.
        forecasts, lower_ci, upper_ci: Pronósticos y sus intervalos de confianza.
        model_name (str): Nombre del modelo para el título.
        fig, axes: Figura y ejes preexistentes (opcional).
    """
    # 1. Creación de la figura y los ejes si no se proporcionan
    if fig is None or axes is None:
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True, 
                                 gridspec_kw={'height_ratios': [3, 1]})

    # --- 2. Subplot 1: Pronóstico vs. Valores Reales ---
    ax1 = axes[0]
    
    ax1.plot(train_t, train_y, label='Datos de Entrenamiento', color='black', linewidth=1.5)
    ax1.plot(test_t, test_y, marker='o', label='Valores Reales (Prueba)', color='darkred', linestyle='', markersize=3)
    ax1.plot(test_t, forecasts, label='Pronóstico Puntual', color='blue', linestyle='--')
    ax1.fill_between(test_t, lower_ci, upper_ci, color='blue', alpha=0.2, label='Intervalo de Confianza (95%)')
    
    ax1.set_title(f'Pronóstico del Modelo {model_name}', fontsize=FONT_SIZES['title'], weight='bold')
    ax1.set_ylabel('Valor de la Serie', fontsize=FONT_SIZES['label'])
    ax1.legend(fontsize=FONT_SIZES['legend'])
    ax1.grid(True, which='both', linestyle=':', linewidth=0.7)
    ax1.tick_params(axis='y', which='major', labelsize=FONT_SIZES['tick'])
    
    # --- 3. Calcular Residuos y Métricas de Error ---
    residuals = test_y - forecasts
    
    # Error Cuadrático Medio (Root Mean Squared Error - RMSE)
    rmse = np.sqrt(np.mean(residuals**2))
    
    # Coeficiente de Determinación (R-cuadrado)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((test_y - np.mean(test_y))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Añadir las métricas al título del subplot de pronóstico
    ax1.set_title(
        f'Pronóstico del Modelo {model_name}\nRMSE: {rmse:.4f} | R²: {r2:.4f}',
        fontsize=FONT_SIZES['title'], weight='bold'
    )
    
    # --- 4. Subplot 2: Análisis de Residuos ---
    ax2 = axes[1]
    
    # Graficar los residuos a lo largo del tiempo
    ax2.stem(test_t, residuals, basefmt="black", linefmt='grey', markerfmt='D')
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
    
    ax2.set_title('Residuos del Pronóstico', fontsize=FONT_SIZES['title']-2)
    ax2.set_xlabel('Fecha', fontsize=FONT_SIZES['label'])
    ax2.set_ylabel('Error', fontsize=FONT_SIZES['label'])
    ax2.grid(True, which='both', linestyle=':', linewidth=0.7)
    ax2.tick_params(axis='both', which='major', labelsize=FONT_SIZES['tick'])
    
    # Mejorar la visualización de las fechas en el eje x
    fig.autofmt_xdate()
    plt.tight_layout(pad=2.0)
    plt.show()

    return fig, axes


def plot_irf(irf_values: np.ndarray, model=None, model_name: str = "ARMA", fig=None, ax=None):
    """
    Plots the Impulse Response Function (IRF) for a time series model.

    Args:
        irf_values (np.ndarray): A 1D array containing the IRF values for each period.
        model_name (str, optional): The name of the model (e.g., "ARMA(2,1)")
                                    to be displayed in the title. Defaults to "ARMA".
        fig (matplotlib.figure.Figure, optional): A pre-existing figure.
        ax (matplotlib.axes.Axes, optional): A pre-existing axis to plot on.
    """
    # 1. Input validation
    if not isinstance(irf_values, np.ndarray) or irf_values.ndim != 1:
        raise TypeError("irf_values must be a 1D numpy array.")
        
    H = len(irf_values)
    periods = np.arange(H)

    # 2. Create the figure and axis if not provided
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    # 3. Create the plot
    ax.plot(periods, irf_values,
            'o-',  # Line with circle markers
            label='Respuesta al Impulso',
            linewidth=2,
            markersize=4,
            color='darkblue')

    # 4. Add a horizontal line at zero
    ax.axhline(0, color='black', linestyle='--', linewidth=1.0)
    
    # 5. Styling and labels
    ax.set_title(f'Función de Impulso-Respuesta (IRF) \n Para Modelo {model_name}({model.q}, {model.p})',
                 fontsize=FONT_SIZES['title'], weight='bold')
    ax.set_xlabel(r'Períodos (Horizonte $H$)', fontsize=FONT_SIZES['label'])
    ax.set_ylabel(r'Respuesta de $y_t$ a un shock de $u_t$', fontsize=FONT_SIZES['label'])
    ax.grid(True, which='both', linestyle=':', linewidth=0.7)
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZES['tick'])
    
    # Set x-axis limits to be tight
    ax.set_xlim(left=-1, right=H)
    
    plt.tight_layout()
    plt.show()

    return fig, ax