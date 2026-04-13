import numpy as np
import numba as nb

from src.quadrature.filon import g, g_prime

@nb.njit(fastmath=True)
def integrand_prop(theta, k0, z_abs, x, R_func):
    """integrand: R(theta) * exp(i * k0 * g(theta))"""
    R_val = R_func(theta)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    return R_val * np.exp(1j * k0 * g(cos_t, sin_t, z_abs, x))
