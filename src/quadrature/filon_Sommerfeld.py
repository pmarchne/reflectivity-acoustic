import sys
import os

#from src.quadrature import filon

import numpy as np
import numba as nb
import time
from scipy.special import roots_legendre


# add src folder to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reflectivity_kx_omega import reflectivity
from layers import to_arrays
from quadrature.filon import precompute_quadrature_points, get_weights_filon, get_weights_filon_numba
from acquisition import Acquisition
from utilities import green2d
from fortran.reflectivity_benchmark import fortran_reflectivity


def integrand_evan_cosh(psi, k0, z_abs, x):
    phase = z_abs *np.sinh(psi) - 1j * x *np.cosh(psi)
    return np.exp(-k0*phase)

@nb.njit(parallel=True, fastmath=True)
def compute_evanescent(dz_vec, dx_vec, k0_vec,
                       sinh_psi, cosh_psi,
                       dz_inverse, rmap_evan, weights_ev, scaling):
    Np = dz_vec.size
    Nw = k0_vec.size
    Nquad = sinh_psi.size
    sinh_psi = sinh_psi.ravel()
    cosh_psi = cosh_psi.ravel()
    weights_ev = weights_ev.ravel()
    k0_vec = k0_vec.ravel()
    acc_evan = np.zeros((Np, Nw), dtype=np.complex128)

    for p in nb.prange(Np):
        ridx = dz_inverse[p]
        dz = dz_vec[p]
        dx = dx_vec[p]
        # Precompute phase arrays (size Nquad)
        phase_min = np.empty(Nquad, dtype=np.complex128)
        phase_plus = np.empty(Nquad, dtype=np.complex128)

        for q in range(Nquad):
            phase_min[q]  = dz * sinh_psi[q] - 1j * dx * cosh_psi[q]
            phase_plus[q] = dz * sinh_psi[q] + 1j * dx * cosh_psi[q]
        # Loop over frequencies
        for w in range(Nw):
            k0 = k0_vec[w]
            s = 0.0 + 0.0j
            for q in range(Nquad):
                val = (
                    np.exp(-k0 * phase_min[q])
                    + np.exp(-k0 * phase_plus[q])
                ) * rmap_evan[ridx, w, q] * weights_ev[q]
                s += val
            acc_evan[p, w] = scaling * s
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

@nb.njit(parallel=True, fastmath=True)
def compute_prop(dz_vec, dx_vec, k0_vec, thetas, Vinv, global_idx, dz_inverse, rmap):
    Np = dz_vec.size
    Nw = k0_vec.size
    Ntheta_eval = len(Vinv)*(len(thetas)-1)
    acc_prop = np.zeros((Np, Nw), dtype=np.complex128)
    for p in nb.prange(Np):
        ridx = dz_inverse[p]
        weights = np.zeros((Nw, Ntheta_eval), dtype=np.complex128)
        weights = get_weights_filon_numba(k0_vec, dz_vec[p], dx_vec[p], thetas, Vinv, global_idx, weights)
        # acc_prop[p, :] = np.sum(weights * rmap, axis=1) # Rmap of size # (Nw, Nquad)
        for w in range(Nw):
            s = 0.0 + 0.0j
            for q in range(Ntheta_eval):
                s += weights[w, q] * rmap[ridx, w, q]
            acc_prop[p, w] = s
    return acc_prop

def Sommerfeld_integral(
    layers,
    omega,
    acq: Acquisition,
    Ntheta, Nevan=64,
    kx_max_factor=4.0, 
    free_surface=1
):
    # layers
    h, vp, _ = to_arrays(layers)
    # omegas
    Nw = omega.size
    k0_vec = omega / vp[0]
    # ---- Acquisition geometry ----
    Ns, Nr = acq.xs.size, acq.xr.size
    
    if len(set(acq.zr)) != 1:
        raise ValueError("All receivers must be at the same depth.")
    if np.any(acq.zr > h[0]):
        raise ValueError(
            f"Receivers cannot be deeper than the top layer depth (h[0] = {h[0]} m)."
        )

    dx_mat = np.abs(acq.xs[:, None] - acq.xr[None, :])  # Ns × Nr
    dz_mat = np.abs(2*h[0] - acq.zs[:, None] - acq.zr[None, :])  # Ns × Nr

    dx_vec = dx_mat.ravel() 
    dz_vec = dz_mat.ravel()
    dz_unique, dz_inverse = np.unique(dz_vec, return_inverse=True)

    Np = dx_vec.size

    # accumulators: (Np, Nwa)
    acc_prop = np.zeros((Np, Nw), dtype=np.complex128)
    acc_evan = np.zeros_like(acc_prop)

    thetas = np.linspace(-np.pi/2., np.pi/2., Ntheta)
    theta_eval, Vinv, global_idx = precompute_quadrature_points(thetas, 'chebychev')

    Nz = len(dz_unique)

    rmap_unique = np.zeros((Nz, Nw, len(theta_eval)), dtype=np.complex128)
    rmap_evan_unique = np.zeros((Nz, Nw, Nevan), dtype=np.complex128)

    p = np.sin(theta_eval) / vp[0]

    sinh_psi, cosh_psi, psi_i, weights_ev, scaling = get_integrand_evan_param(kx_max_factor, Nevan)
    ph = np.cosh(psi_i) / vp[0]

    for i, dzi in enumerate(dz_unique):
        zs_eff = 2.*h[0] - acq.zr[0] - dzi # since dz = |2h - zr - zs|
        rmap_unique[i] = fortran_reflectivity(
            layers, omega, p,
            free_surface=free_surface,
            zr=acq.zr[0],
            zs=zs_eff
        )
        rmap_evan_unique[i] = fortran_reflectivity(
            layers, omega, ph,
            free_surface=free_surface,
            zr=acq.zr[0],
            zs=zs_eff
        )

    #rmap = fortran_reflectivity(layers, omega, p, free_surface=1, zr=76., zs=76.)
    #rmap_evan = fortran_reflectivity(layers, omega, ph, free_surface=1, zr=76., zs=76.)

    acc_prop = compute_prop(dz_vec, dx_vec, k0_vec, thetas, Vinv, global_idx, dz_inverse, rmap_unique)
    acc_evan = compute_evanescent(dz_vec, dx_vec, k0_vec, sinh_psi, cosh_psi, dz_inverse, rmap_evan_unique, weights_ev, scaling)

    I_total = -(acc_evan+1j*acc_prop) / (4.*np.pi)
    res_pairs = I_total.reshape((Ns, Nr, Nw))

    return res_pairs

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    layers = [
        (100.0, 1500.0, 2000.0),
        (250.0, 2900.0, 2000.0),
    ]
    h, vp = layers[0][0], layers[0][1]

    freqs = np.linspace(0.1, 300.0, 600)
    omega = 2.0 * np.pi * freqs #+ 10.*1j
    # x and z positions of sources
    sources = [(30.0, 76.0), (80.0, 76.0)]
    receivers = [(200.0, 76.0), (800.0, 76.0)]
    acq = Acquisition(sources, receivers)
    xs, zs, xr, zr = acq.xs, acq.zs, acq.xr, acq.zr

    r_direct = np.sqrt( (acq.xs[0] - acq.xr[0])**2 + (acq.zs[0] - acq.zr[0])**2 )

    kx_factor = 4.
    nq_prop = 600
    nq_evan = 256
    start = time.time() # z_travel
    result = Sommerfeld_integral(
        layers, omega, xs, zs, xr, zr,
        nq_prop, nq_evan, kx_max_factor=kx_factor)
    end = time.time()
    print(f"quadrature elapsed: {end-start:.2f} s")

    G_quad = result[0, 0, :]
    Green = green2d(omega, vp, r_direct)

    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(omega, np.real(Green), 'r--')
    plt.plot(omega, np.real(G_quad), 'b:')
    plt.xlabel('omega')
    plt.grid()

    plt.subplot(1,2,2)
    plt.plot(omega, np.imag(Green), 'g--')
    plt.plot(omega, np.imag(G_quad), 'b-.')
    plt.xlabel('omega')
    plt.grid()
    plt.show()

