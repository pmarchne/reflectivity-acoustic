import numpy as np
import time
from src.parameters import Parameters
from src.acquisition import Acquisition
from src.utilities import source_frequency, green2d_flat, inverse_fft_signal
# from src.quadrature.gauss_cheby import integral_kx_quadrature_numba
from src.quadrature.filon_Sommerfeld import Sommerfeld_integral

def forward(layers,
            acq: Acquisition,
            param: Parameters,
            free_surface=False,
            nq_prop=1024):
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

    # 1. generate frequency domain array and source wavelet
    source_freq, omegas, delay = source_frequency(param)
    # 2. quadrature in spatial Fourier domain (kx or incidence angles)
    start = time.time()
    R_map = Sommerfeld_integral(
        layers, omegas, acq,
        nq_prop, Nevan=64, kx_max_factor=4., free_surface=free_surface)
    end = time.time()
    print(f"kx quadrature elapsed: {end-start:.2f} s")

    # flatten for all source-receiver pairs
    Ns, Nr, Nw = R_map.shape
    R_flat = R_map.reshape((Ns*Nr, Nw))

    # add Green function in homogeneous medium (top layer)
    transfer = R_flat + green2d_flat(omegas, layers[0][1], acq.get_distances())
    # apply source delay
    transfer_delayed = transfer \
        * np.exp(-1j * delay * omegas)[None, :]

    # convolution with source in the frequency domain
    response = transfer_delayed * source_freq[None, :] 
    # T_flat *= 1j*np.real(omegas) # ricker source time-derivative !
    traces = inverse_fft_signal(response, param)
    d_cal = traces.reshape((Ns, Nr, param.nt))
    return d_cal
