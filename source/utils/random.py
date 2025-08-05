import numpy as np
from numpy.polynomial.polynomial import polyfromroots


def generate_stationary_phi(p: int) -> np.ndarray:
    """
    Genera coeficientes phi para un proceso AR(p) estacionario garantizado.
    """
    # Calculamos cuántos pares de raíces complejas necesitamos
    num_complex_pairs = p // 2
    
    # Generamos los componentes aleatorios para todos los pares a la vez
    # Radios al cuadrado (uniformes en [0,1]) para evitar sesgo hacia el centro
    radii_sq = np.random.uniform(0, 1, size=num_complex_pairs)
    radii = np.sqrt(radii_sq)

    # Ángulos uniformes en [0, 2*pi]
    thetas = np.random.uniform(0, 2 * np.pi, size=num_complex_pairs)
    
    # Creamos los pares de raíces complejas y sus conjugadas
    complex_roots = radii * np.exp(1j * thetas)
    roots = np.concatenate([complex_roots, np.conjugate(complex_roots)])

    # Si p es impar, añadimos una raíz real entre -1 y 1
    if p % 2 != 0:
        real_root = np.random.uniform(-1, 1, size=1)
        roots = np.concatenate([roots, real_root])

    # --- 2. Construir el polinomio y extraer los coeficientes phi ---
    
    # Construimos el polinomio a partir de las raíces: c0, c1, ..., c(p-1), 1
    poly_coeffs = polyfromroots(roots)
    
    # Los coeficientes phi son los negativos de los coeficientes del polinomio
    # (excepto el último que es 1) en orden inverso.
    # Se toma la parte real para descartar ruido numérico flotante.
    phi = -np.real(poly_coeffs[:-1][::-1])
    
    return phi.reshape(1, p)