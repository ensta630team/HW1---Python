import numpy as np 
import sympy as sp

def calculate_irf(F, H, phi=None, method='exact'):
    p = F.shape[-1]

    irf_values = np.zeros(H)
    irf_values[0] = 1

    eigenvals, eigenvec = np.linalg.eig(F)
    
    # check eigenvalues (we can use schur function from scipy CHECK!)
    mult = len(set(eigenvals))
    if len(eigenvals) != mult  or method=='jordan':
        Fsymp = sp.Matrix(F)
        P, J = Fsymp.jordan_form()
        print(J)

    elif method == 'exact' and len(eigenvals) == mult:
        for h in range(1, H):
            temp = np.eye(p)*np.pow(eigenvals, h)
            temp = np.matmul(eigenvec, temp)
            temp = np.matmul(temp, np.linalg.inv(eigenvec))
            irf_values[h] = temp[0, 0]

    elif method == 'exact_old' and len(eigenvals) == mult:
        for h in range(1, H):
            F_h = np.linalg.matrix_power(F, h)
            irf_values[h] = F_h[0, 0]

    if method == 'simulation':
        for h in range(1, H):
            # Get the last p values of the IRF generated so far
            past_values = irf_values[max(0, h - F.shape[-1]):h]
            # The coefficients to use depend on how many past values we have
            coeffs_to_use = phi[:len(past_values)]
            irf_values[h] = np.dot(coeffs_to_use, past_values[::-1])
    
    return irf_values