"""
Seismic simulation: forward modeling and adjoint gradient computation.

Typical usage
-------------
Build once* from a class src.config.Config, then call as many times as
needed inside an inversion loop:

    sim = Simulation(config)

    # Compute synthetic seismogram
    d_cal, cache = sim.forward(layers)

    # Compute parameter gradients (FWI / gradient-based inversion)
    residual = d_cal - d_obs
    grad_vp, grad_rho = sim.gradient(residual[0], layers, cache)
"""

import numpy as np
import numba as nb

from src.config import Config
from src.builders import build_problem
from src.utilities import (
    source_frequency,
    inverse_fft_signal,
    adjoint_inverse_fft_signal,
    timer,
)
from src.kernels import green2d
from src.quadrature.filon_Sommerfeld import Sommerfeld_integral2D
from src.fortran.reflectivity_adjoint import fortran_reflectivity_adj


class Simulation:
    """Forward and adjoint seismic simulation.

    Parameters
    ----------
    config : Config
        Experiment configuration (geometry, numerics, source).

    Examples
    --------
    Forward-only:
        sim = Simulation(config)
        d_cal, _ = sim.forward(layers)
    Forward + gradient (FWI loop)::
        sim = Simulation(config)
        d_cal, cache   = sim.forward(layers)
        grad_vp, _     = sim.gradient(residual[0], layers, cache)
    """

    def __init__(self, config: Config):
        self.config = config
        self.param, self.acq = build_problem(config)
        self._source_freq = source_frequency(self.param, config)

    def forward(self, layers, timing: bool = False) -> tuple[np.ndarray, dict]:
        """Compute synthetic seismogram for a given earth model.

        Parameters
        ----------
        layers : list of (h, vp, rho) tuples
            1-D earth model.
        timing : bool
            If True, print wall-clock time of the Sommerfeld quadrature.

        Returns
        -------
        d_cal : ndarray, shape (Ns, Nr, Nt)
            Synthetic seismogram in the time domain.
        cache : dict
            Intermediate arrays required by gradient.
        """
        return _forward(
            layers, self.config, self.param, self.acq, self._source_freq, timing
        )

    def gradient(self, residual, layers, cache) -> tuple[np.ndarray, np.ndarray]:
        """Compute parameter gradients via the adjoint state method.

        Parameters
        ----------
        residual : ndarray, shape (Nr, Nt)
            Time-domain data residual 'd_cal - d_obs' for **one source**.
        layers : list of (h, vp, rho) tuples
        cache : dict
            Intermediate arrays returned by 'forward' call.

        Returns
        -------
        grad_vp : ndarray, shape (n_layers,)
            Gradient of the misfit w.r.t. P-wave velocity.
        grad_rho : ndarray, shape (n_layers,)
            Gradient of the misfit w.r.t. density.
        """
        return _gradient(
            residual, layers, self._source_freq, self.config, self.param, cache
        )


def _forward(
    layers, config, param, acq, source_freq, timing
) -> tuple[np.ndarray, dict]:
    vp_top = layers[0][1]

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

    Ns, Nr, Nw = green_multi.shape
    green_multi = green_multi.reshape((Ns * Nr, Nw))

    dist_direct = acq.distances_direct()
    green_multi += green2d(param.omegas, vp_top, dist_direct) # add direct wave contribution
    if config.free_surface:
        # with free surface, add contribution from the ghost source
        dist_ghost = acq.distances_ghost()
        green_multi -= green2d(param.omegas, vp_top, dist_ghost)

    response = green_multi * source_freq[None, :]
    if config.source_deriv:
        response *= 1j * np.real(param.omegas)

    d_cal = inverse_fft_signal(response, param, config)
    return d_cal.reshape((Ns, Nr, param.nt)), cache


def _gradient(residual, layers, source_freq, config, param, cache):
    adj_response = adjoint_inverse_fft_signal(residual, param, config)
    if config.source_deriv:
        adj_response *= -1j * np.real(param.omegas)

    adj_green = adj_response * np.conj(source_freq)
    adj_acc_evan = -adj_green / (4.0 * np.pi)
    adj_acc_prop = 1j * adj_green / (4.0 * np.pi)

    adj_R_prop = _accum_adjoint(adj_acc_prop, cache["weights_prop"])
    adj_R_evan = _accum_adjoint(adj_acc_evan, cache["kernel_evan"])

    _, dR_dvp_prop, dR_drho_prop = fortran_reflectivity_adj(
        layers,
        param.omegas,
        cache["p_prop"],
        free_surface=config.free_surface,
        zr=config.z_rec,
        zs=config.z_src,
    )
    _, dR_dvp_evan, dR_drho_evan = fortran_reflectivity_adj(
        layers,
        param.omegas,
        cache["p_evan"],
        free_surface=config.free_surface,
        zr=config.z_rec,
        zs=config.z_src,
    )

    grad_vp = _sum_gradient(adj_R_prop, dR_dvp_prop, adj_R_evan, dR_dvp_evan)
    grad_rho = _sum_gradient(adj_R_prop, dR_drho_prop, adj_R_evan, dR_drho_evan)

    grad_vp[0] = 0.0  # top layer held fixed
    grad_rho[0] = 0.0
    return grad_vp, grad_rho


def _sum_gradient(seed_prop, dR_dm_prop, seed_evan, dR_dm_evan):
    res_prop = np.einsum("wq,wql->l", np.conj(seed_prop), dR_dm_prop, optimize=True)
    res_evan = np.einsum("wq,wql->l", np.conj(seed_evan), dR_dm_evan, optimize=True)
    return np.real(res_prop + res_evan)


@nb.njit(parallel=True, fastmath=True)
def _accum_adjoint(adj_acc, weights):
    Np, Nw, Nq = weights.shape
    out = np.zeros((Nw, Nq), dtype=np.complex128)
    for w in nb.prange(Nw):
        row = np.zeros(Nq, dtype=np.complex128)
        for p in range(Np):
            a = adj_acc[p, w]
            for q in range(Nq):
                row[q] += np.conj(weights[p, w, q]) * a
        out[w, :] = row
    return out
