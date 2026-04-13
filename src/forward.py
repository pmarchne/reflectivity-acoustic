import numpy as np

from src.config import Config
from src.builders import build_problem
from src.utilities import source_frequency, inverse_fft_signal, timer
from src.kernels import green2d
from src.quadrature.filon_Sommerfeld import Sommerfeld_integral2D


class ForwardSimulation:
    """
    Bundles a fixed Config with its pre-built Parameters and Acquisition.

    Build once, call run(layers) as many times as needed.
    This is the preferred pattern for inversion loops where the
    acquisition geometry and numerical parameters are fixed.

    """

    def __init__(self, config: Config):
        self.config = config
        self.param, self.acq = build_problem(config)

    def run(self, layers, timing: bool = False) -> tuple[np.ndarray, dict]:
        """Run the forward model for the given earth model.

        Returns:
            d_cal : (Ns, Nr, nt) time-domain seismogram
            cache : intermediate arrays needed by the adjoint
        """
        return _run_forward(layers, self.config, self.param, self.acq, timing)


def _run_forward(layers, config, param, acq, timing) -> tuple[np.ndarray, dict]:
    """Core forward computation used by ForwardSimulation.run()."""
    vp_top = layers[0][1]  # P-wave velocity of the top (water) layer

    # 1. Source wavelet in the frequency domain
    source_freq = source_frequency(param, config)

    # 2. Sommerfeld integral over incidence angles
    with timer("Sommerfeld quadrature", timing):
        green_multi, cache = Sommerfeld_integral2D(
            layers,
            param.omegas,
            acq,
            config.nq_prop,
            config.nq_evan,
            kx_max_factor=config.kx_max_factor,
            free_surface=config.free_surface,
        )

    # Flatten source-receiver pairs for vectorized operations
    Ns, Nr, Nw = green_multi.shape
    green_multi = green_multi.reshape((Ns * Nr, Nw))

    # 3. Add direct-wave Green's function; subtract free-surface ghost if needed
    dist_direct = acq.distances_direct()
    green_multi += green2d(param.omegas, vp_top, dist_direct)
    if config.free_surface:
        dist_ghost = acq.distances_ghost()
        green_multi -= green2d(param.omegas, vp_top, dist_ghost)

    # 4. Convolve with source wavelet in the frequency domain
    response = green_multi * source_freq[None, :]
    if config.source_deriv:
        response *= 1j * np.real(param.omegas)  # time-derivative of Ricker source

    # 5. Inverse FFT back to the time domain
    d_cal = inverse_fft_signal(response, param, config)

    return d_cal.reshape((Ns, Nr, param.nt)), cache
