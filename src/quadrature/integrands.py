import numpy as np
import os
import sys

import numba as nb

# Add src folder to Python path if running from an outer directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

''' 
-------------------- 2D Sommerfeld Integral --------------------
Propagative regime : after the substitution kx = k0 sin(theta), the 2D Sommerfeld integral takes the form
            I_prop(x,z,w) = 1/(2*i)*int_{-pi/2}^{pi/2} R(theta, w) e^{i k_0 * g(theta)}
with 
    - k0 = w / vp_{top}
    - R(theta, w): the reflectivity map
    - g(theta) = |z|*cos(theta) + x*sin(theta)

Evanescent regime : after the substitution s = k0 cosh(psi), the 2D Sommerfeld integral takes the form
            I_evan(x,z,w) = -1/(4*pi)*int_{0}^{psi_max} R(psi, w) e^{-k_0 * h(psi)}
with 
    - k0 = w / vp_{top}
    - R(psi, w): the reflectivity map
    - h(psi) = |z|*sinh(psi) + i*x*cosh(psi)
'''

@nb.njit(fastmath=True)
def g(cos_t, sin_t, z_abs, x):
    """Phase function g(theta) = |z|*cos(theta) + x*sin(theta)"""
    return z_abs * cos_t + x * sin_t

@nb.njit(fastmath=True)
def g_prime(cos_t, sin_t, z_abs, x):
    """Derivative g'(theta) = -|z|*sin(theta) + x*cos(theta)"""
    return -z_abs * sin_t + x * cos_t

@nb.njit(fastmath=True)
def integrand_prop(theta, k0, z_abs, x, R_func):
    """integrand: R(theta) * exp(i * k0 * g(theta))"""
    R_val = R_func(theta)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    return R_val * np.exp(1j * k0 * g(cos_t, sin_t, z_abs, x))
