import numpy as np
from scipy.special import hankel1

def ricker_wavelet(t, f0):
    """
    Parameters:
        t : float or np.ndarray
            Time variable.
        f0 : float
            Characteristic frequency in Hz.
    Returns:
        w : np.ndarray
            Wavelet values.
        t0 : float
            Time shift applied.
    """
    t0 = 0.1  # time shift [s]
    tau = t - t0
    w = (1 - 2 * (np.pi * f0 * tau)**2) * np.exp(-(np.pi * f0 * tau)**2)
    return w, t0

def green2d(omega, c, r):
    """
    2D Green's function for a point source in homogeneous medium.
    Parameters:
        omega : float
            Angular frequency (rad/s).
        c : float
            Wave speed.
        r : float
            Distance from the source.
    Returns:
        G : complex
            Green's function value at distance r.
    """
    k = omega / c
    return -(1j / 4) * hankel1(0, k * r)

def green2d_flat(omegas, c, distances):
    """
    Vectorized 2D Green's function for multiple frequencies and receiver distances.
    Parameters:
        omegas : array-like
            Angular frequencies (rad/s).
        c : float
            Wave speed.
        distances : array-like
            Distances from source to receivers.
    Returns:
        G : np.ndarray
            Complex Green's function matrix of shape (N_receivers, N_frequencies).
    """
    omegas = np.asarray(omegas)
    r = np.ravel(distances).copy()
    r[r == 0] = 1e-10  # avoid singularity
    kr = np.outer(omegas / c, r)  # shape (N_frequencies, N_receivers)
    G = -(1j / 4) * hankel1(0, kr)  # shape (N_frequencies, N_receivers)
    return G.T  # shape (N_receivers, N_frequencies)

def get_kz_chunk(omega, c, kx_chunk) -> np.ndarray:
    """
    Compute vertical wavenumber kz for a chunk of horizontal wavenumbers.
    Parameters:
        omega : array-like, shape (Nw,)
            Angular frequencies.
        c : float
            Wave speed for the layer.
        kx_chunk : array-like, shape (chunk,)
            Horizontal wavenumbers for this chunk.
    Returns:
        kz : np.ndarray, shape (Nw, chunk)
            Vertical wavenumber matrix (principal branch, imag(kz) >= 0).
    """
    omega = np.asarray(omega)  # shape (Nw,)
    kx = np.asarray(kx_chunk)[None, :]  # shape (1, chunk)
    k0 = omega[:, None] / c  # shape (Nw, 1)

    kz2 = k0**2 - kx**2  # shape (Nw, chunk)
    kz = np.sqrt(kz2 + 0j)  # principal branch

    # enforce principal branch (imag(kz) >= 0)
    kz = np.where(np.imag(kz) < 0, -kz, kz)
    return kz

