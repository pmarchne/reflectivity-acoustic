import sys
import os

#from src.quadrature import filon

import numpy as np
import numba as nb
import time
from scipy.special import roots_legendre


# add src folder to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#from reflectivity_kx_omega import reflectivity
from layers import to_arrays
from quadrature.filon import precompute_quadrature_points, get_weights_filon, get_weights_filon_numba
from acquisition import Acquisition
from fortran.reflectivity_benchmark import fortran_reflectivity, reflectivity_q
from utilities import get_kz

def integrand_evan_cosh(psi, k0, z_abs, x):
    phase = z_abs *np.sinh(psi) - 1j * x *np.cosh(psi)
    return np.exp(-k0*phase)

@nb.njit(parallel=True, fastmath=True)
def compute_evanescent(dz_vec, dx_vec, k0_vec,
                       sinh_psi, cosh_psi,
                       rmap_evan, weights_ev, scaling):
    Np = dz_vec.size
    Nw = k0_vec.size
    Nquad = sinh_psi.size

    sinh_psi = sinh_psi.ravel()
    cosh_psi = cosh_psi.ravel()
    weights_ev = weights_ev.ravel()
    
    k0_vec = k0_vec.ravel()
    acc_evan = np.zeros((Np, Nw), dtype=np.complex128)
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
            s = 0.0 + 0.0j
            for q in range(Nquad):
                exp_min = np.exp(-k0 * phase_min[q])
                exp_plus = np.exp(-k0 * phase_plus[q])
                kernel = scaling * (exp_min + exp_plus) * weights_ev[q]
                kernel_evan[p, w, q] = kernel # store kernel for adjoint
                s += kernel * rmap_evan[w, q]
                #val = (
                #    np.exp(-k0 * phase_min[q])
                #    + np.exp(-k0 * phase_plus[q])
                #) * rmap_evan[w, q] * weights_ev[q]
                #s += val
            acc_evan[p, w] = s #scaling * s
    return acc_evan, kernel_evan

def get_integrand_evan_param(kx_max_factor, nevan):
    psi_max = np.arccosh(kx_max_factor)
    # Gauss-Legendre quadrature points in [0, psi_max]
    pts, weights_leg = roots_legendre(nevan)
    scale_factor = psi_max / 2.0
    psi_i = scale_factor * (pts + 1.0)  # Map from [-1,1] to [0, psi_max]
    sinh_psi = np.sinh(psi_i[None, :])
    cosh_psi = np.cosh(psi_i[None, :])
    return sinh_psi, cosh_psi, psi_i, weights_leg, scale_factor

@nb.njit(parallel=True, fastmath=True)
def compute_prop(dz_vec, dx_vec, k0_vec, thetas, Vinv, global_idx, rmap):
    Np = dz_vec.size
    Nw = k0_vec.size
    Ntheta_eval = len(Vinv)*(len(thetas)-1)

    acc_prop = np.zeros((Np, Nw), dtype=np.complex128)
    weights_prop = np.zeros((Np, Nw, Ntheta_eval), dtype=np.complex128)

    for p in nb.prange(Np):
        weights = np.zeros((Nw, Ntheta_eval), dtype=np.complex128)
        weights = get_weights_filon_numba(k0_vec, dz_vec[p], dx_vec[p], thetas, Vinv, global_idx, weights)
        weights_prop[p, :, :] = weights # store weights for adjoint
        for w in range(Nw):
            s = 0.0 + 0.0j
            for q in range(Ntheta_eval):
                s += weights[w, q] * rmap[w, q]
            acc_prop[p, w] = s
    return acc_prop, weights_prop


def Sommerfeld_integral2D(
    layers,
    omega,
    acq: Acquisition,
    Ntheta, Nevan=64,
    kx_max_factor=4.0, 
    free_surface=1
):
    # layers properties
    h, vp, _ = to_arrays(layers)
    Nw = omega.size
    Ns, Nr = acq.xs.size, acq.xr.size
    
    if len(set(acq.zr)) != 1:
        raise ValueError("All receivers must be at the same depth.")
    if np.any(acq.zr > h[0]):
        raise ValueError(
            f"Receivers cannot be deeper than the top layer depth (h[0] = {h[0]} m)."
        )

    dx_mat = np.abs(acq.xs[:, None] - acq.xr[None, :])  # Ns × Nr
    dx_vec = dx_mat.ravel()
    Np = dx_vec.size

    # quadrature setup
    thetas = np.linspace(-np.pi/2., np.pi/2., Ntheta)
    theta_eval, Vinv, global_idx = precompute_quadrature_points(thetas, 'chebychev')
    sinh_psi, cosh_psi, psi_i, weights_ev, scaling = get_integrand_evan_param(kx_max_factor, Nevan)
    p = np.sin(theta_eval) / vp[0]
    ph = np.cosh(psi_i) / vp[0]

    # reflectivity of the stack
    #R_prop = reflectivity_q(layers, omega, p)
    R_prop = fortran_reflectivity(layers, omega, p, free_surface=free_surface, zr=acq.zr[0], zs=acq.zs[0])
    R_evan = fortran_reflectivity(layers, omega, ph, free_surface=free_surface, zr=acq.zr[0], zs=acq.zs[0])
    #R_evan = reflectivity_q(layers, omega, ph)
    #kz0_prop, kz0_evan = get_kz(omega, vp[0], p), get_kz(omega, vp[0], ph) 

    if free_surface:
        dz_refl = 2.0 * h[0]
    else:
        dz_refl = 2.0 * h[0] - acq.zs[0] - acq.zr[0]

    # accumulators: (Np, Nwa)
    acc_prop = np.zeros((Np, Nw), dtype=np.complex128)
    acc_evan = np.zeros_like(acc_prop)
    k0_vec = omega / vp[0]
    dz_vec = np.full_like(dx_vec, dz_refl)

    acc_prop, weights_prop = compute_prop(
        dz_vec, dx_vec, k0_vec, thetas, Vinv, global_idx, R_prop
    )
    acc_evan, kernel_evan = compute_evanescent(
        dz_vec, dx_vec, k0_vec, sinh_psi, cosh_psi,
        R_evan, weights_ev, scaling
    )

    cache = {
        "weights_prop": weights_prop,
        "kernel_evan": kernel_evan,
        "p_prop": p,
        "p_evan": ph,
    }

    int_total = -(acc_evan+1j*acc_prop) / (4.*np.pi)
    res_pairs = int_total.reshape((Ns, Nr, Nw))

    return res_pairs, cache

