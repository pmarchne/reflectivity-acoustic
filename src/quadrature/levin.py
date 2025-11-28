import sys
import os

# add src folder to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.linalg import solve

# --- Helper Functions (Assumed from previous context) ---
def g(theta, z_abs, x):
    """Phase function g(theta) = |z|*cos(theta) + x*sin(theta)"""
    return z_abs * np.cos(theta) + x * np.sin(theta)

def g_prime(theta, z_abs, x):
    """Derivative g'(theta) = -|z|*sin(theta) + x*cos(theta)"""
    return -z_abs * np.sin(theta) + x * np.cos(theta)

def basisfunctions_radial(u):
    """ this basis is taken from the article 
    Niskanen, M., & Lähivaara, T. (2023). COMPOSTI: A Python-based program for seismic trans-dimensional inversion. SoftwareX, 21, 101298.
    which is surprisingly stable when solving for the Levin system !
    """
    n = len(u)
    centerpoints = u - 0.25 * (u[-1] - u[0])
    d0 = (u[-1] - u[0]) / 2.0 
    eps = 1.0 / d0 
    basis = np.zeros((n, n))
    basisp = np.zeros_like(basis)
    for k in range(0, n):
        r = u - centerpoints[k]
        # Basis function: phi_k(u) = sqrt(1 + (eps * r_k(u))^2)
        basis[:, k] = np.sqrt(1 + (eps * r)**2)
        # Basis derivative: phi_k'(u) = eps^2 * r_k(u) / sqrt(1 + (eps * r_k(u))^2)
        basisp[:, k] = eps**2 * r / np.sqrt(1 + (eps * r)**2)
    return basis, basisp

def evaluate_P(u_val, centers, eps):
    n = len(centers)
    row = np.zeros(n)
    for k in range(n):
        r = u_val - centers[k]
        row[k] = np.sqrt(1 + (eps * r)**2)
    return row
    
# --- Levin Scheme ---
def levin_mq_quad(a, b, k0, z_abs, x, R_func, N=12, epsilon=1.0):
    """
    Levin Quadrature on [a, b] using Multiquadric RBFs and Chebyshev points.

    Parameters:
        a, b (float): Integration limits.
        k0 (float): Wave number.
        z_abs, x (float): Parameters for the phase function g(theta).
        R_func (callable): The amplitude function R(theta).
        N (int): Number of Chebyshev collocation points. Default is 12.
        epsilon (float): MQ RBF shape parameter. Default is 1.0 (requires tuning!).
        
    Returns:
        complex: The integral approximation on [a, b].
    """
    
    # Collocation Points (Chebyshev-Gauss-Lobatto)
    cheb_nodes = -np.cos(np.linspace(0, np.pi, N))
    theta_j = (b - a) * (cheb_nodes + 1) / 2.0 + a
    phi, phi_prime = basisfunctions_radial(theta_j)
    
    R_vec = R_func(theta_j) # The right-hand side vector R
    g_prime_j = g_prime(theta_j, z_abs, x)

    # Levin equation: P' + i*k0*g'*P = R
    A = np.zeros((N, N), dtype=complex)
    D_g_prime = np.diag(1j * k0 * g_prime_j)
    A = phi_prime + D_g_prime @ phi

    try:
        alpha = solve(A, R_vec)
    except np.linalg.LinAlgError:
        print("Levin MQ Solver Failed: Matrix is singular. Check basis definition/parameters.")
        return 0.0 + 0.0j
    
    d0_temp = (theta_j[-1] - theta_j[0]) / 2.0
    eps_temp = 1.0 / d0_temp
    centerpoints_temp = theta_j - 0.25 * (theta_j[-1] - theta_j[0])
    
    Phi_a = evaluate_P(a, centerpoints_temp, eps_temp)
    Phi_b = evaluate_P(b, centerpoints_temp, eps_temp)
    P_a = Phi_a @ alpha
    P_b = Phi_b @ alpha
    
    # Calculate integral based on boundary terms
    exp_a = np.exp(1j * k0 * g(a, z_abs, x))
    exp_b = np.exp(1j * k0 * g(b, z_abs, x))
    
    I_approx = (P_b * exp_b) - (P_a * exp_a)
    
    return I_approx

def composite_levin(A, B, M, k0, z_abs, x, R_func, N=12, epsilon=1.0):
    points = np.linspace(A, B, M + 1)
    total_integral = 0.0 + 0.0j
    for i in range(M):
        a_i, b_i = points[i], points[i+1]
        subinterval_integral = levin_mq_quad(
            a_i, b_i, k0, z_abs, x, R_func, N=N, epsilon=epsilon
        )
        total_integral += subinterval_integral
    return total_integral