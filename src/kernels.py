import numpy as np
from scipy.special import hankel1


def green2d(omegas, c, distances, eps=1e-10):
    """
    2D outgoing Green's function for the Helmholtz equation
    using the exp(-i * w * t) time-harmonic convention.

    G(r, w) = -(i/4) H_0^(1)(k r),   with k = w / c

    Parameters
    ----------
    omegas : float or array-like
        Angular frequency/frequencies [rad/s].
    c : float
        Wave speed [m/s].
    distances : float or array-like
        Source-receiver distance(s) [m].
    eps : float, optional
        Small positive value used to regularize singularities at r = 0
        and kr = 0 (default is 1e-10).

    Returns
    -------
    G : complex ndarray or complex
        Green's function evaluated at all receiver/frequency pairs.
        Shape is (n_receivers, n_frequencies) if inputs are arrays,
        or a scalar if both inputs are scalars.
    """
    omegas = np.atleast_1d(omegas)
    r = np.atleast_1d(distances).astype(float)

    # Regularize r = 0 to avoid singularities
    r = np.maximum(r, eps)

    k = omegas / c
    kr = np.outer(k, r)

    # Regularize kr = 0 for Hankel evaluation
    kr = np.maximum(kr, eps)

    g2d = -(1j / 4) * hankel1(0, kr)

    if g2d.size == 1:
        return g2d.item()

    return g2d.T  # (n_receivers, n_frequencies)


def green3d(omegas, c, distances, eps=1e-10):
    """
    3D outgoing Green's function for the Helmholtz equation
    using the exp(-i * w * t) time-harmonic convention.

    G(r, w) = exp(i k r) / (4 pi r),   with k = w / c

    Parameters
    ----------
    omegas : float or array-like
        Angular frequency/frequencies [rad/s].
    c : float
        Wave speed [m/s].
    distances : float or array-like
        Source-receiver distance(s) [m].
    eps : float, optional
        Small positive value used to regularize the singularity at r = 0
        (default is 1e-10).

    Returns
    -------
    G : complex ndarray or complex
        Green's function evaluated at all receiver/frequency pairs.
        Shape is (n_receivers, n_frequencies) if inputs are arrays,
        or a scalar if both inputs are scalars.
    """
    omegas = np.atleast_1d(omegas)
    r = np.atleast_1d(distances).astype(float)

    # Regularize r = 0 to avoid singularity
    r = np.maximum(r, eps)

    k = omegas / c
    kr = np.outer(k, r)

    g3d = np.exp(1j * kr) / (4 * np.pi * r[None, :])

    if g3d.size == 1:
        return g3d.item()

    return g3d.T  # (n_receivers, n_frequencies)
