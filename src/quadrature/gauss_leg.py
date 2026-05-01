import numpy as np
from scipy.special import roots_legendre


def get_weights(a, b, npts):
    t, weights = roots_legendre(npts)
    jac = (b - a) / 2.0
    points = jac * t + (b + a) / 2.0
    return weights, points, jac


def gauss_legendre_quad(a, b, npts, k0, z_abs, x, R, integrand):
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
    t, w = roots_legendre(npts)
    pts = 0.5*(b - a)*t + 0.5*(b + a)
    F = integrand(pts[:, None], k0, z_abs, x, R)
    return 0.5*(b - a) * (w @ F)
