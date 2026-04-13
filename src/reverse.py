import numpy as np
import numba as nb
from src.config import Config
from src.builders import build_problem
from src.utilities import adjoint_inverse_fft_signal, source_frequency
from src.fortran.reflectivity_adjoint import fortran_reflectivity_adj


class ReverseSimulation:
    """
    Bundles a fixed Config with its pre-built Parameters for the adjoint (reverse) simulation.

    Build once, call run(residual, layers, cache) as many times as needed.
    This is the preferred pattern for inversion loops where the
    acquisition geometry and numerical parameters are fixed.
    """

    def __init__(self, config: Config):
        self.config = config
        self.param, self.acq = build_problem(config)
        self._source_freq = source_frequency(self.param, config)

    def run(self, residual, layers, cache) -> tuple[np.ndarray, np.ndarray]:
        """Backpropagate residuals through the L2 misfit to obtain parameter gradients.

        Parameters:
            residual    : (Nr, Nt) time-domain residual for one source
            layers      : earth model (list of (z, vp, rho) tuples)
            cache       : intermediate arrays returned by ForwardSimulation.run()

        Returns:
            grad_vp  : gradient w.r.t. P-wave velocity, shape (n_layers,)
            grad_rho : gradient w.r.t. density, shape (n_layers,)
        """
        return _compute_gradient(
            residual, layers, self._source_freq, self.config, self.param, cache
        )


def _compute_gradient(residual, layers, source_freq, config, param, cache):
    """Core adjoint computation used by ReverseSimulation.run()."""
    # Adjoint FFT: residual shape (Nr, Nt) for one source
    adj_response = adjoint_inverse_fft_signal(residual, param, config)
    if config.source_deriv:
        adj_response *= -1j * np.real(param.omegas)

    # Back through source multiplication
    adj_green = adj_response * np.conj(source_freq)

    # Back through Sommerfeld split
    adj_acc_evan = -adj_green / (4.0 * np.pi)
    adj_acc_prop = 1j * adj_green / (4.0 * np.pi)

    # Back through quadrature sums
    weights_prop = cache["weights_prop"]
    kernel_evan = cache["kernel_evan"]

    adj_R_prop = compute_prop_adjoint_numba(adj_acc_prop, weights_prop)
    adj_R_evan = compute_evanescent_adjoint_numba(adj_acc_evan, kernel_evan)

    # Back through reflectivity
    p = cache["p_prop"]
    ph = cache["p_evan"]

    _, dR_dvp_prop, dR_drho_prop = fortran_reflectivity_adj(
        layers,
        param.omegas,
        p,
        free_surface=config.free_surface,
        zr=config.z_rec,
        zs=config.z_src,
    )
    _, dR_dvp_evan, dR_drho_evan = fortran_reflectivity_adj(
        layers,
        param.omegas,
        ph,
        free_surface=config.free_surface,
        zr=config.z_rec,
        zs=config.z_src,
    )

    grad_vp = sum_gradient(adj_R_prop, dR_dvp_prop, adj_R_evan, dR_dvp_evan)
    grad_rho = sum_gradient(adj_R_prop, dR_drho_prop, adj_R_evan, dR_drho_evan)

    # Top layer is held fixed
    grad_vp[0] = 0.0
    grad_rho[0] = 0.0

    return grad_vp, grad_rho


def sum_gradient(seed_prop, dR_dm_prop, seed_evan, dR_dm_evan):
    """Contract adjoint seeds with reflectivity Jacobians to form the gradient."""
    res_prop = np.einsum(
        "wq,wql->l",
        np.conj(seed_prop),
        dR_dm_prop,
        optimize=True,
    )
    res_evan = np.einsum(
        "wq,wql->l",
        np.conj(seed_evan),
        dR_dm_evan,
        optimize=True,
    )
    return np.real(res_prop + res_evan)


@nb.njit(parallel=True, fastmath=True)
def compute_prop_adjoint_numba(adj_acc_prop, weights_prop):
    Np, Nw, Nq = weights_prop.shape
    adj_R_prop = np.zeros((Nw, Nq), dtype=np.complex128)

    for w in nb.prange(Nw):
        row = np.zeros(Nq, dtype=np.complex128)
        for p in range(Np):
            a = adj_acc_prop[p, w]
            for q in range(Nq):
                row[q] += np.conj(weights_prop[p, w, q]) * a
        adj_R_prop[w, :] = row

    return adj_R_prop


@nb.njit(parallel=True, fastmath=True)
def compute_evanescent_adjoint_numba(adj_acc_evan, kernel_evan):
    Np, Nw, Nq = kernel_evan.shape
    adj_R_evan = np.zeros((Nw, Nq), dtype=np.complex128)

    for w in nb.prange(Nw):
        row = np.zeros(Nq, dtype=np.complex128)
        for p in range(Np):
            a = adj_acc_evan[p, w]
            for q in range(Nq):
                row[q] += np.conj(kernel_evan[p, w, q]) * a
        adj_R_evan[w, :] = row

    return adj_R_evan
