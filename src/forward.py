import numpy as np
import time
from src.parameters import Parameters
from src.acquisition import Acquisition
from src.utilities import ricker_wavelet, green2d_flat
from src.quadrature.gauss_cheby import integral_kx_quadrature_numba

def forward(layers, acq: Acquisition, param: Parameters, free_surface=False, nq_prop=1024, nq_evan=512):
    """
    Forward modeling to compute predicted data d_cal
    using the reflectivity method with optional free surface.
    
    Args:
        layers: list of tuples (thickness, vp, rho)
        acq: Acquisition object with xs, zs, xr, zr
        param: Parameters object with dt, nfft, nt, time, freq
        free_surface: whether to include free surface at z=0
    
    Returns:
        d_cal: array (Ns, Nr, nt) of predicted time-domain traces
    """
    freqs = np.fft.rfftfreq(param.nfft, param.dt)
    epsilon = 1.5 # complex frequency damping
    omegas = 2.0 * np.pi * freqs + 1j * epsilon

    s_t, delay = ricker_wavelet(param.time, param.freq)
    s_t = s_t * np.exp(-epsilon * param.time)
    print(f"Ricker wavelet initialized with delay: {delay} s")
    # ---- FFT in omega using the +i w t convention ----
    s_w = np.conj(np.fft.rfft(np.conj(s_t), n=param.nfft)) 
    vp_top = layers[0][1] # velocity of the top layer

    # ---- Acquisition geometry ----
    xs, zs, xr, zr = acq.xs, acq.zs, acq.xr, acq.zr
    Ns, Nr = xs.size, xr.size

    h_top = layers[0][0]
    # Two-way vertical travel
    if free_surface:
        h_reflector = 0.0
        # image source at z=0
        z_travel = 2.0 * (zs[:, None] - 0.0) + zr[None, :]
    else:
        h_reflector = h_top
        # first interface below top layer
        z_travel = 2.*(h_top - zs[:, None]) + zr[None, :]
    
    z_travel = np.abs(h_reflector - zs[:, None]) + np.abs(h_reflector - zr[None, :])

    # Pairwise horizontal distances
    travel_xy = np.sqrt((xs[:, None] - xr[None, :])**2 + (zs[:, None] - zr[None, :])**2)

    # ---- kx quadrature ----
    start = time.time()
    R_map = integral_kx_quadrature_numba(
        layers, omegas, xs, zs, xr, z_travel,
        nq_prop, nq_evan, fs=free_surface)
    end = time.time()
    print(f"kx quadrature elapsed: {end-start:.2f} s")

    # ---- Flatten for source-receiver pairs ----
    Ns, Nr, Nw = R_map.shape
    R_flat = R_map.reshape((Ns*Nr, Nw))
    distances_flat = travel_xy.ravel()
    G_flat = green2d_flat(omegas, vp_top, distances_flat)

    T_flat = (G_flat+R_flat) * s_w[None, :] * np.exp(-1j * delay * omegas)[None, :]
    # ---- Inverse FFT to time domain ----
    traces_full = np.conj(np.fft.irfft(np.conj(T_flat), n=param.nfft, axis=1))
    traces_cut = traces_full[:, :param.nt] * np.exp(epsilon * param.time)

    # ---- Reshape to (Ns, Nr, nt) ----
    d_cal = traces_cut.reshape((Ns, Nr, param.nt))
    return d_cal