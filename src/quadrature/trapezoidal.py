import numpy as np


def trapezoidal_quad(a, b, n, k0, z_abs, x, R, integrand):
    pts = np.linspace(a, b, n)
    F = integrand(pts[:, None], k0, z_abs, x, R)
    return np.trapz(F, pts, axis=0)