import numpy as np
from src.parameters import Parameters
from src.acquisition import Acquisition
from src.utilities import (
    source_frequency,
    inverse_fft_signal,
    timer,
)
from src.kernels import green2d
from src.quadrature.filon_Sommerfeld import Sommerfeld_integral2D


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
    source_freq, omegas = source_frequency(param)
    # 2. quadrature in spatial Fourier domain (incidence angles)
    with timer("Sommerfeld quadrature", timing):
        green_multi = Sommerfeld_integral2D(
            layers, omegas, acq,
            nq_prop, Nevan=128, kx_max_factor=4., free_surface=free_surface)

    # flatten for all source-receiver pairs
    Ns, Nr, Nw = green_multi.shape
    green_multi = green_multi.reshape((Ns*Nr, Nw))

    # 3. add Green function in homogeneous medium (top layer)
    dist_direct = acq.get_distances()
    if not free_surface:
        green_multi += green2d(omegas, layers[0][1], dist_direct)
    else:
        # Calculate distances for the direct surface ghost
        dx_mat = np.abs(acq.xs[:, None] - acq.xr[None, :])
        dz_ghost_mat = acq.zs[:, None] + acq.zr[None, :]
        dist_ghost = np.sqrt(dx_mat**2 + dz_ghost_mat**2).ravel()
        
        # Add primary direct wave, subtract the phase-flipped direct ghost
        green_multi += green2d(omegas, layers[0][1], dist_direct)
        green_multi -= green2d(omegas, layers[0][1], dist_ghost)

    # reverse seismic source delay back to t=0 sec
    #green_multi *= np.exp(-1j * 0.75*delay * omegas)[None, :]

    # 4. convolution with source in the frequency domain
    response = green_multi * source_freq[None, :]
    response *= 1j*np.real(omegas) # ricker source time-derivative !

    # 5. Inverse FFT to go back in time
    d_cal = inverse_fft_signal(response, param)

    return d_cal.reshape((Ns, Nr, param.nt))
