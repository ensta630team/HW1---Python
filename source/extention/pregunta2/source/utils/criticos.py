import numpy as np

def valores_criticos(estadisticos: list[float]):
    # Ordenamos el vector de menor a mayor
    estadisticos = np.sort(estadisticos)
    # Calculo
    valor_critico_1 = np.percentile(estadisticos, 99)
    valor_critico_5 = np.percentile(estadisticos, 95)
    valor_critico_10 = np.percentile(estadisticos, 90)

    print("Valor crítico al 1%:", valor_critico_1)
    print("Valor crítico al 5%:", valor_critico_5)
    print("Valor crítico al 10%:", valor_critico_10)