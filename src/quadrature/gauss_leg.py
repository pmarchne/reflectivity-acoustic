import numpy as np
from scipy.special import roots_legendre


def get_weights(a, b, npts):
    t, weights = roots_legendre(npts)
    jac = (b - a) / 2.0
    points = jac * t + (b + a) / 2.0
    return weights, points, jac


# def gauss_legendre_quad(theta_q, weights, jac, k0, z_abs, x, r_map, integrand):
#    F_i = integrand(theta_q, k0, z_abs, x, r_map)
#    integral_approx = jac * np.sum(weights * F_i)
#    return integral_approx
