import numpy as np


def integrand(kx, k0, z_abs, x, r_func):
    """integrand: R(theta) * exp(i * k0 * g(theta))"""
    r_val = r_func(kx)
    kz = np.sqrt(k0**2 - kx**2 + 0j)
    den = 2.*1j*kz
    return (r_val / den) * np.exp(1j * kz * z_abs)  * np.exp(1j * kx * x)
