import numpy as np

from src.config import Config
from src.builders import build_problem
from src.utilities import source_frequency, inverse_fft_signal, timer
from src.kernels import green2d
from src.quadrature.filon_Sommerfeld import Sommerfeld_integral2D


def forward(layers, config: Config, timing: bool = False):
    """
    Forward modeling to compute predicted data d_cal using the reflectivity method.

    Args:
        layers: list of Layer objects (or tuples of (thickness, vp, rho))
        config: Config object with all modeling parameters
        timing: if True, print elapsed time for the Sommerfeld quadrature step
    Returns:
        d_cal: array (Ns, Nr, nt) of predicted time-domain traces
        cache: dict of intermediate arrays used by the adjoint
    """
    param, acq = build_problem(config)

    # 1. Generate source wavelet in the frequency domain
    source_freq = source_frequency(param, config)

    # 2. Sommerfeld integral over incidence angles
    with timer("Sommerfeld quadrature", timing):
        green_multi, cache = Sommerfeld_integral2D(
            layers,
            param.omegas,
            acq,
            config.nq_prop,
            config.nq_evan,
            kx_max_factor=4.0,
            free_surface=config.free_surface,
        )

    # Flatten source-receiver pairs for vectorised operations
    Ns, Nr, Nw = green_multi.shape
    green_multi = green_multi.reshape((Ns * Nr, Nw))

    # 3. Add direct-wave Green's function in the homogeneous top layer
    dist_direct = acq.distances_direct()
    if not config.free_surface:
        green_multi += green2d(param.omegas, layers[0][1], dist_direct)
    else:
        dist_ghost = acq.distances_ghost()
        # Primary direct wave minus the phase-flipped free-surface ghost
        green_multi += green2d(param.omegas, layers[0][1], dist_direct)
        green_multi -= green2d(param.omegas, layers[0][1], dist_ghost)

    # 4. Convolve with source wavelet in the frequency domain
    response = green_multi * source_freq[None, :]
    if config.source_deriv:
        response *= 1j * np.real(param.omegas)  # time-derivative of Ricker source

    # 5. Inverse FFT back to the time domain
    d_cal = inverse_fft_signal(response, param, config)

    return d_cal.reshape((Ns, Nr, param.nt)), cache
