import numpy as np
import numba as nb
from scipy.special import roots_legendre


def integrand_evan_cosh(psi, k0, z_abs, x):
    phase = z_abs * np.sinh(psi) - 1j * x * np.cosh(psi)
    return np.exp(-k0 * phase)

@nb.njit(parallel=True, fastmath=True)
def get_kernel_evan(
    dz_vec, dx_vec, k0_vec, sinh_psi, cosh_psi, weights_ev, scaling
):
    Np = dz_vec.size
    Nw = k0_vec.size
    Nquad = sinh_psi.size
    sinh_psi = sinh_psi.ravel()
    cosh_psi = cosh_psi.ravel()
    weights_ev = weights_ev.ravel()
    k0_vec = k0_vec.ravel()
    kernel_evan = np.zeros((Np, Nw, Nquad), dtype=np.complex128)

    for p in nb.prange(Np):
        dz = dz_vec[p]
        dx = dx_vec[p]
        # Precompute phase arrays (size Nquad)
        phase_min = np.empty(Nquad, dtype=np.complex128)
        phase_plus = np.empty(Nquad, dtype=np.complex128)

        for q in range(Nquad):
            phase_min[q] = dz * sinh_psi[q] - 1j * dx * cosh_psi[q]
            phase_plus[q] = dz * sinh_psi[q] + 1j * dx * cosh_psi[q]
        # Loop over frequencies
        for w in range(Nw):
            k0 = k0_vec[w]
            for q in range(Nquad):
                exp_min = np.exp(-k0 * phase_min[q])
                exp_plus = np.exp(-k0 * phase_plus[q])
                kernel = scaling * (exp_min + exp_plus) * weights_ev[q]
                kernel_evan[p, w, q] = kernel  # store kernel for adjoint
    return kernel_evan


@nb.njit(parallel=True, fastmath=True)
def compute_evanescent(rmap_evan, kernel_evan):
    Np, Nw, Nquad = kernel_evan.shape
    acc_evan = np.zeros((Np, Nw), dtype=np.complex128)
    for p in nb.prange(Np):
        # Loop over frequencies
        for w in range(Nw):
            s = 0.0 + 0.0j
            for q in range(Nquad):
                s += kernel_evan[p, w, q] * rmap_evan[w, q]
            acc_evan[p, w] = s  # scaling * s
    return acc_evan


def get_integrand_evan_param(kx_max_factor, nevan):
    psi_max = np.arccosh(kx_max_factor)
    # Gauss-Legendre quadrature points in [0, psi_max]
    pts, weights_leg = roots_legendre(nevan)
    scale_factor = psi_max / 2.0
    psi_i = scale_factor * (pts + 1.0)  # Map from [-1,1] to [0, psi_max]
    sinh_psi = np.sinh(psi_i[None, :])
    cosh_psi = np.cosh(psi_i[None, :])
    return sinh_psi, cosh_psi, psi_i, weights_leg, scale_factor