
import sys
import os

# add src folder to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from reflectivity_kx_omega import _reflectivity_numba_core
from layers import layers_to_arrays
from numpy.polynomial.chebyshev import chebgauss
from numpy.polynomial.legendre import leggauss
from numba import njit, prange


def integral_kx_quadrature_numba(
    layers,
    omega,
    xs, zs, xr, zr,
    n_prop,
    n_evan,
    kx_max_factor=2.0,
    chunk=256,
    fs=False
):
    """
    Numba-accelerated integral over kx for ALL source-receiver pairs.

    Returns acc_pairs shaped (Np, Nwa), where Np = Ns*Nr and Nwa = number of active omegas.
    """
    # layers
    h, vp, rho = layers_to_arrays(layers)

    # ---- Acquisition geometry ----
    xs = np.asarray(xs).ravel()
    zs = np.asarray(zs).ravel()
    xr = np.asarray(xr).ravel()
    zr = np.asarray(zr).ravel()
    Ns, Nr = xs.size, xr.size
    r_mat = np.abs(xs[:, None] - xr[None, :])
    dz_mat = np.abs(zs[:, None] - zr[None, :])
    r_vec = r_mat.ravel()
    dz_vec = dz_mat.ravel()
    Np = r_vec.size

    # omegas
    omega = np.atleast_1d(omega)
    nz_mask = (omega != 0.0)
    if not np.any(nz_mask):
        # return zeros with full Nw size if desired
        return np.zeros((Np, omega.size))

    omega_act = omega[nz_mask]         # (Nwa,)
    Nwa = omega_act.size

    # quadrature nodes
    nodes_cheb, weights_cheb = chebgauss(n_prop)
    nodes_leg, weights_leg = leggauss(n_evan)
    u_all = nodes_cheb[nodes_cheb >= 0]
    w_u_all = weights_cheb[nodes_cheb >= 0]

    alpha = float(kx_max_factor)
    nodes_leg_scaled = (0.5 * (alpha - 1.0) * nodes_leg + 0.5 * (alpha + 1.0))
    w_leg_scaled = 0.5 * (alpha - 1.0) * weights_leg

    k0 = omega_act / layers[0][1]  # (Nwa,)
    # accumulators: (Np, Nwa)
    acc_prop = np.zeros((Np, Nwa), dtype=np.complex128)
    acc_evan = np.zeros_like(acc_prop)

    # loop chunks for propagating (q = u_all)
    n_u = u_all.size
    for i0 in range(0, n_u, chunk):
        i1 = min(i0 + chunk, n_u)
        u_chunk = u_all[i0:i1]
        w_chunk = w_u_all[i0:i1]

        # build kx_chunk and kz_chunk arrays with shape (Nwa, chunk)
        # kx = k0[:,None] * u_chunk[None,:]
        kx_chunk = (k0[:, None] * u_chunk[None, :]) 
        kz_chunk = (k0[:, None] * np.sqrt(np.clip(1.0 - u_chunk[None, :]**2, 0.0, None)))

        # reflectivity (numba core): expects kx_chunk as complex (Nwa,chunk)
        R_chunk = _reflectivity_numba_core(h, vp, rho, omega_act, kx_chunk, free_surface=fs)
        # call propagating accumulator (in-place add)
        _accumulate_prop_numba(k0, kx_chunk, kz_chunk, R_chunk, w_chunk, r_vec, dz_vec, acc_prop)

    # loop chunks for evanescent (q in [1,alpha])
    n_leg = nodes_leg_scaled.size
    for i0 in range(0, n_leg, chunk):
        i1 = min(i0 + chunk, n_leg)
        q_chunk = nodes_leg_scaled[i0:i1]
        w_chunk = w_leg_scaled[i0:i1]

        kx_chunk = (k0[:, None] * q_chunk[None, :])
        sqrt_term = np.sqrt(np.clip(q_chunk[None, :]**2 - 1.0, 0.0, None))
        kz_chunk = (1j * k0[:, None] * sqrt_term)

        R_chunk = _reflectivity_numba_core(h, vp, rho, omega_act, kx_chunk, free_surface=fs)
        # pass jacobian as vector of ones if weights already scaled
        _accumulate_evan_numba(k0, kx_chunk, kz_chunk, R_chunk, w_chunk, r_vec, dz_vec, acc_evan, k0)

    acc_total = acc_prop + acc_evan   # (Np, Nwa)
    acc_total *= -1j / (2.0 * np.pi)
    # reconstruct full omega size with zeros for omega==0
    res_flat = np.zeros((Np, omega.size), dtype=np.complex128)
    res_flat[:, nz_mask] = acc_total
    # reshape to (Ns, Nr, Nw) if desired by caller, or return flattened (Np,Nw)
    res_pairs = res_flat.reshape((Ns, Nr, omega.size))
    return res_pairs   # shape (Ns, Nr, Nw)

# ---------- Numba accumulator for propagating part ----------
@njit(parallel=True, fastmath=True)
def _accumulate_prop_numba(k0, kx_chunk, kz_chunk, R_chunk, w_chunk, r_vec, dz_vec, out_acc):
    """
    Accumulate propagating integral for one chunk.
    k0: (Nwa,) real
    kx_chunk, kz_chunk, R_chunk: (Nwa, chunk)
    w_chunk: (chunk,) real
    r_vec, dz_vec: (Np,) real
    out_acc: (Np, Nwa) complex128 accumulator (in-place add)
    """
    Nwa = k0.shape[0]
    chunk = kx_chunk.shape[1]
    Np = r_vec.shape[0]

    for p in prange(Np):
        r = r_vec[p]
        dz = dz_vec[p]
        for iw in range(Nwa):
            acc = 0.0 + 0.0j
            # local references (faster)
            kx_row = kx_chunk[iw]
            kz_row = kz_chunk[iw]
            R_row = R_chunk[iw]
            for j in range(chunk):
                kx = kx_row[j]
                kz = kz_row[j]
                Rv = R_row[j]
                # compute integrand: exp(i kz dz) * cos(kx*r) * R
                val = np.exp(1j * kz * dz) * np.cos(kx * r) * Rv #* np.exp(1j * kz * (dz))
                acc += val * w_chunk[j]
            out_acc[p, iw] += acc #/ k0[iw]

# ---------- Numba accumulator for evanescent part ----------
@njit(parallel=True, fastmath=True)
def _accumulate_evan_numba(k0, kx_chunk, kz_chunk, R_chunk, w_chunk, r_vec, dz_vec, out_acc, jacobian_freq):
    """
    Accumulate evanescent integral for one chunk.
    jacobian_chunk: (chunk,) real factor associated to each kx sample (may include k0 factor if needed)
    out_acc: (Np, Nwa) complex accumulator (in-place add)
    """
    Nwa = k0.shape[0]
    chunk = kx_chunk.shape[1]
    Np = r_vec.shape[0]

    for p in prange(Np):
        r = r_vec[p]
        dz = dz_vec[p]
        for iw in range(Nwa):
            acc = 0.0 + 0.0j
            kx_row = kx_chunk[iw]
            kz_row = kz_chunk[iw]
            R_row = R_chunk[iw]
            for j in range(chunk):
                kx = kx_row[j]
                kz = kz_row[j]
                Rv = R_row[j]
                val = (1.0 / kz) * np.exp(1j * kz * dz) * np.cos(kx * r) * Rv
                acc += val * w_chunk[j]
            out_acc[p, iw] += acc * jacobian_freq[iw]

            
''' attempt to include FS
    h0: top interface depth
    s_val = np.exp(1j * kz * (z_ref - zs_p))  # downgoing from source to interface
    Pr_val = np.exp(1j * kz * (z_ref - zr_p))   # upgoing from interface to receiver
    #Gd_val = np.exp(1j * kz * (zr_p - zs_p))            # direct vertical
    T_val = Pr_val * Rv * Ps_val

    T_ghost = 0.0 + 0.0j
    R_fs = -1.0  # Pressure release boundary
    Ps_fs = np.exp(1j * kz * zs_p)  # up to FS
    Pr_fs = np.exp(1j * kz * zr_p)  # down from FS
    T_ghost = Ps_fs * R_fs * Pr_fs
    
    Ps_fs_to_int = np.exp(1j * kz * (zs_p + h0))  # source→FS→interface
    Pr_int_to_rcv = np.exp(1j * kz * (h0 - zr_p))  # interface→receiver
    T_fs_reflected = Ps_fs_to_int * R_fs * Rv * Pr_int_to_rcv
                     
    Ps_to_int = np.exp(1j * kz * (h0 - zs_p))  # source→interface
    Pr_fs_to_rcv = np.exp(1j * kz * (h0 + zr_p))  # interface→FS→receiver
    T_reflected_fs = Ps_to_int * Rv * R_fs * Pr_fs_to_rcv
                    
    T_ghost += T_fs_reflected + T_reflected_fs
'''
    
if __name__ == "__main__":
    layers = [
        (100.0, 1500.0, 1800.0),
        (250.0, 1900.0, 2000.0),
        (350.0, 1700.0, 2200.0),
        (500.0, 2000.0, 2400.0),
    ]

    xs = 20.
    zs = 0.
    xr = 100.
    zr = 0.
    n_prop = 1024
    n_evan = 512

    freqs = np.linspace(0.1, 100.0, 1024)
    omega = 2.0 * np.pi * freqs

    import time
    start = time.time()
    res = integral_kx_quadrature_numba(layers, omega, xs, zs, xr, zr, n_prop, n_evan)
    end = time.time()
    print(f"kx quadrature elapsed: {end-start:.2f} s")
    print(res.shape)

    xs = np.array([20., 220., 420., 620.])
    xr = np.array([50., 150., 250., 350., 450., 550, 650.])
    zs = np.array([0., 0., 0., 0.])
    zr = np.array([0., 0., 0., 0., 0., 0., 0.])
    start = time.time()
    res = integral_kx_quadrature_numba(layers, omega, xs, zs, xr, zr, n_prop, n_evan)
    end = time.time()
    print(f"kx quadrature elapsed: {end-start:.2f} s")