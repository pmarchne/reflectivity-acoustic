import numpy as np
from src.parameters import Parameters
from src.acquisition import Acquisition
from src.utilities import (
    source_frequency,
    green2d_flat,
    inverse_fft_signal,
    timer,
)
from src.quadrature.filon_Sommerfeld import Sommerfeld_integral

def forward(layers,
            acq: Acquisition,
            param: Parameters,
            free_surface=False,
            nq_prop=1024, timing=False):
    """
    Forward modeling to compute predicted data d_cal
    using the reflectivity method.

    Args:
        layers: list of tuples (thickness, vp, rho)
        acq: Acquisition object with xs, zs, xr, zr
        param: Parameters object with dt, nfft, nt, time, freq
        free_surface (optional): include free surface at z=0
    Returns:
        d_cal: array (Ns, Nr, nt) of predicted time-domain traces
    """

    # 1. generate frequency domain array and source wavelet
    source_freq, omegas, delay = source_frequency(param)
    # 2. quadrature in spatial Fourier domain (kx or incidence angles)
    with timer("Sommerfeld quadrature", timing):
        rmap = Sommerfeld_integral(
            layers, omegas, acq,
            nq_prop, Nevan=256, kx_max_factor=4., free_surface=free_surface)

    # flatten for all source-receiver pairs
    Ns, Nr, Nw = rmap.shape
    rmap = rmap.reshape((Ns*Nr, Nw))

    # 3. add Green function in homogeneous medium (top layer)
    transfer = rmap \
        + green2d_flat(omegas, layers[0][1], acq.get_distances())
    # reverse seismic source delay back to t=0 sec
    transfer_delayed = transfer \
        * np.exp(-1j * delay * omegas)[None, :]

    # 4. convolution with source in the frequency domain
    response = transfer_delayed * source_freq[None, :] 
    # T_flat *= 1j*np.real(omegas) # ricker source time-derivative !

    # 5. Inverse FFT to go back in time
    d_cal = inverse_fft_signal(response, param)

    return d_cal.reshape((Ns, Nr, param.nt))
