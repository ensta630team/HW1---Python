import numpy as np
from numpy.polynomial.polynomial import polyfromroots


DISTRIBUTION_GENERATORS = {
    'normal': lambda **kwargs: np.random.normal(
        loc=kwargs.get('mean', 0.0),      # Default mean = 0
        scale=kwargs.get('std', 1.0),       # Default std = 1
        size=kwargs['p']),
    'exponential': lambda **kwargs: np.random.exponential(
        scale=kwargs.get('scale', 1.0),   # Default scale = 1
        size=kwargs['p']),
    'uniform': lambda **kwargs: np.random.uniform(
        low=kwargs.get('low', 0.0),       # Default low = 0
        high=kwargs.get('high', 1.0),     # Default high = 1
        size=kwargs['p']),
    'beta': lambda **kwargs: np.random.beta(
        a=kwargs.get('alpha', 1.0),       # Default alpha = 1
        b=kwargs.get('beta', 1.0),        # Default beta = 1
        size=kwargs['p']),
    'gamma': lambda **kwargs: np.random.gamma(kwargs['shape'], scale=kwargs.get('scale', 1.0), size=kwargs['p']),
    'poisson': lambda **kwargs: np.random.poisson(lam=kwargs.get('lambda', 1.0), size=kwargs['p'])
}

def generate_stationary_phi(p: int) -> np.ndarray:
    num_complex_pairs = p // 2
    radii_sq = np.random.uniform(0, 1, size=num_complex_pairs)
    radii = np.sqrt(radii_sq)
    thetas = np.random.uniform(0, 2 * np.pi, size=num_complex_pairs)
    complex_roots = radii * np.exp(1j * thetas)
    roots = np.concatenate([complex_roots, np.conjugate(complex_roots)])
    if p % 2 != 0:
        real_root = np.random.uniform(-1, 1, size=1)
        roots = np.concatenate([roots, real_root])
    poly_coeffs = polyfromroots(roots)
    phi = -np.real(poly_coeffs[:-1][::-1])
    return phi.reshape(1, p)

def initialize_params(params_distribution, **kwargs):
    if params_distribution is None:
        return None

    if isinstance(params_distribution, (list, np.ndarray)):
        return np.array(params_distribution)

    if params_distribution == 'stationary':
        return generate_stationary_phi(kwargs.get('p', 2)).flatten()

    generator_func = DISTRIBUTION_GENERATORS.get(params_distribution)

    if generator_func:
        try:
            return generator_func(**kwargs)
        except KeyError as e:
            raise TypeError(f"Missing required keyword argument: {e} for distribution '{params_distribution}'")
    else:
        raise ValueError(f"'{params_distribution}' is not a valid value for params_distribution.")
    
def get_windows(serie: np.ndarray, n_ventanas: int, tamano_ventana: int) -> np.ndarray:
    largo_serie = len(serie)
    if tamano_ventana > largo_serie:
        raise ValueError("El tamaño de la ventana no puede ser mayor que el largo de la serie.")
    max_ventanas_posibles = largo_serie - tamano_ventana + 1
    

    posibles_inicios = np.arange(max_ventanas_posibles)
    
    inicios_elegidos = np.random.choice(
        a=posibles_inicios,
        size=n_ventanas,
        replace=True
    )

    indices_ventanas = inicios_elegidos[:, np.newaxis] + np.arange(tamano_ventana)
    ventanas = serie[indices_ventanas]
    
    return ventanas