import numpy as np
from src.config import Config
from src.builders import build_problem
from src.utilities import (
    source_frequency,
    inverse_fft_signal,
    timer,
)
from src.kernels import green2d
from src.quadrature.filon_Sommerfeld import Sommerfeld_integral2D


def forward(layers,
            config: Config,
            timing=False):
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
    param, acq = build_problem(config)
    # 1. generate frequency domain array and source wavelet
    source_freq = source_frequency(param, config)
    # 2. quadrature in spatial Fourier domain (incidence angles)
    with timer("Sommerfeld quadrature", timing):
        green_multi, cache = Sommerfeld_integral2D(
            layers, param.omegas, acq,
            config.nq_prop, config.nq_evan,
            kx_max_factor=4., free_surface=config.free_surface)

    # flatten for all source-receiver pairs
    Ns, Nr, Nw = green_multi.shape
    green_multi = green_multi.reshape((Ns*Nr, Nw))

    # 3. add Green function in homogeneous medium (top layer)
    dist_direct = acq.distances_direct()
    if not config.free_surface:
        green_multi += green2d(param.omegas, layers[0][1], dist_direct)
    else:
        dist_ghost = acq.distances_ghost()
        # Add primary direct wave, subtract the phase-flipped direct ghost
        green_multi += green2d(param.omegas, layers[0][1], dist_direct)
        green_multi -= green2d(param.omegas, layers[0][1], dist_ghost)

    # reverse seismic source delay back to t=0 sec
    # green_multi *= np.exp(-1j * 0.75*delay * omegas)[None, :]

    # 4. convolution with source in the frequency domain
    response = green_multi * source_freq[None, :]
    
    if config.source_deriv:
        response *= 1j*np.real(param.omegas)  # ricker source time-derivative !

    # 5. Inverse FFT to go back in time
    d_cal = inverse_fft_signal(response, param, config)

    return d_cal.reshape((Ns, Nr, param.nt)), cache
