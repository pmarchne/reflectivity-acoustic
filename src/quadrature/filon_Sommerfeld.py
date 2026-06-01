import numpy as np
import numba as nb

from src.layers import to_arrays
from src.quadrature.filon import precompute_quadrature_points, get_weights_filon_numba
from src.quadrature.quad_evanescent import get_integrand_evan_param, compute_evanescent, get_kernel_evan
from src.acquisition import Acquisition
from src.fortran.reflectivity_benchmark import fortran_reflectivity
from src.utilities import timer


@nb.njit(parallel=True, fastmath=True)
def get_weights_prop(dz_vec, dx_vec, k0_vec, thetas, Vinv, global_idx):
    Np = dz_vec.size
    Nw = k0_vec.size
    Ntheta_eval = len(Vinv) * (len(thetas) - 1)
    weights_prop = np.zeros((Np, Nw, Ntheta_eval), dtype=np.complex128)
    for p in nb.prange(Np):
        weights = np.zeros((Nw, Ntheta_eval), dtype=np.complex128)
        weights = get_weights_filon_numba(
            k0_vec, dz_vec[p], dx_vec[p], thetas, Vinv, global_idx, weights
        )
        weights_prop[p, :, :] = weights
    return weights_prop


@nb.njit(parallel=True, fastmath=True)
def compute_prop(dz_vec, k0_vec, thetas, Vinv, rmap, weights_prop):
    Np = dz_vec.size
    Nw = k0_vec.size
    Ntheta_eval = len(Vinv) * (len(thetas) - 1)
    acc_prop = np.zeros((Np, Nw), dtype=np.complex128)
    for p in nb.prange(Np):
        weights = weights_prop[p, :, :] # read stored weights
        for w in range(Nw):
            s = 0.0 + 0.0j
            for q in range(Ntheta_eval):
                s += weights[w, q] * rmap[w, q]
            acc_prop[p, w] = s
    return acc_prop


def Sommerfeld_integral2D(
    layers,
    omega,
    acq: Acquisition,
    Ntheta,
    Nevan=64,
    kx_max_factor=4.0,
    free_surface: bool = True,
    cache: dict = {}
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
    thetas = np.linspace(-np.pi / 2.0, np.pi / 2.0, Ntheta)
    theta_eval, Vinv, global_idx = precompute_quadrature_points(thetas, "chebychev")
    
    sinh_psi, cosh_psi, psi_i, weights_ev, scaling = get_integrand_evan_param(
        kx_max_factor, Nevan
    )

    # accumulators: (Np, Nwa)
    acc_prop = np.zeros((Np, Nw), dtype=np.complex128)
    acc_evan = np.zeros_like(acc_prop)
    k0_vec = omega / vp[0]

    if free_surface:
        dz_refl = 2.0 * h[0] # with free surface, the total path length is twice the depth of the top layer
    else:
        dz_refl = 2.0 * h[0] - acq.zs[0] - acq.zr[0] # without free surface, the path length is the sum of source and receiver depths
    dz_vec = np.full_like(dx_vec, dz_refl)

    if cache:
        weights_prop = cache["weights_prop"]
        kernel_evan = cache["kernel_evan"]
        p_unique = cache["p_unique"]
        inv_idx = cache["inv_idx"]
        ph = cache["p_evan"]
    else:
        p = np.sin(theta_eval) / vp[0]
        ph = np.cosh(psi_i) / vp[0]
        # since R(theta) is an even function, we only need to compute it for positive p and then mirror the values for negative p.
        p_rounded = np.round(np.abs(p), decimals=15)
        p_unique, inv_idx = np.unique(p_rounded, return_inverse=True)
        # check if shape of p is consistent with shape of weights_prop
        if len(p_unique) != len(p)//2:
            raise ValueError(f"Shape of p ({len(p_unique)}) does not match expected shape")
        
        weights_prop = get_weights_prop(dz_vec, dx_vec, k0_vec, thetas, Vinv, global_idx)
        kernel_evan = get_kernel_evan(dz_vec, dx_vec, k0_vec, sinh_psi, cosh_psi, weights_ev, scaling)

        cache.update({
            "weights_prop": weights_prop,
            "kernel_evan": kernel_evan,
            "p_unique": p_unique,
            "inv_idx": inv_idx,
            "p_evan": ph,
        })

    # reflectivity of the stack
    R_prop_unique = fortran_reflectivity(
        layers, omega, p_unique, free_surface=free_surface, zr=acq.zr[0], zs=acq.zs[0]
    )
    R_prop = R_prop_unique[:, inv_idx]
    R_evan = fortran_reflectivity(
        layers, omega, ph, free_surface=free_surface, zr=acq.zr[0], zs=acq.zs[0]
    )

    acc_prop = compute_prop(
        dz_vec, k0_vec, thetas, Vinv, R_prop, weights_prop
    )
    acc_evan = compute_evanescent(R_evan, kernel_evan)

    int_total = -(acc_evan + 1j * acc_prop) / (4.0 * np.pi)
    res_pairs = int_total.reshape((Ns, Nr, Nw))

    return res_pairs
