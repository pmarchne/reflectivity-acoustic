import numpy as np
from scipy.special import roots_legendre


def get_weights(a, b, npts):
    t, weights = roots_legendre(npts)
    jac = (b - a) / 2.0
    points = jac * t + (b + a) / 2.0
    return weights, points, jac


def gauss_legendre_quad(a, b, npts, k0, z_abs, x, R_func, integrand):
    """
    Gauss-Legendre quadrature of `integrand` over [a, b].

    Parameters
    ----------
    a, b      : integration limits
    npts      : number of quadrature points
    k0        : wave number at the current frequency
    z_abs     : |z| depth parameter
    x         : horizontal offset
    R_func    : reflectivity callable R(kx_or_psi)
    integrand : callable(nodes, k0, z_abs, x, R_func) → array

    Returns
    -------
    scalar complex approximation of the integral
    """
    weights, points, jac = get_weights(a, b, npts)
    F = integrand(points, k0, z_abs, x, R_func)
    return jac * np.sum(weights * F)
