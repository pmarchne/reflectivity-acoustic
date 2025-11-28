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
from scipy.special import y0, j0


def integral_kx_quadrature_numba(
    layers,
    omega,
    xs, zs, xr, zr,
    n_prop,
    n_evan,
    kx_max_factor=4.0,
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

    dz_mat = np.abs(2*h[0] - zr[:, None] - zs[None, :])
    dz_mat_plus = np.abs(zr[:, None] + zs[None, :])
    dz_mat_minus = np.abs(zr[:, None] - zs[None, :])
    r_vec = r_mat.ravel()
    dz_vec = dz_mat.ravel()
    dz_plus = dz_mat_plus.ravel()
    dz_minus = dz_mat_minus.ravel()
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

    k0 = omega_act / vp[0]  # (Nwa,)
    # accumulators: (Np, Nwa)
    acc_prop = np.zeros((Np, Nwa), dtype=np.complex128)
    acc_evan = np.zeros_like(acc_prop)

    hd = h[0]
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
        R_chunk = _reflectivity_numba_core(h, vp, rho, omega_act, kx_chunk, 0., 0., False)
        # call propagating accumulator (in-place add)
        _accumulate_prop_numba(k0, kx_chunk, kz_chunk, R_chunk, w_chunk, r_vec, dz_vec, dz_plus, dz_minus, hd, acc_prop, free_surface=fs)#, fs, zs, zr)

    # loop chunks for evanescent (q in [1,alpha])
    n_leg = nodes_leg_scaled.size
    for i0 in range(0, n_leg, chunk):
        i1 = min(i0 + chunk, n_leg)
        q_chunk = nodes_leg_scaled[i0:i1]
        w_chunk = w_leg_scaled[i0:i1]

        kx_chunk = (k0[:, None] * q_chunk[None, :])
        kz_chunk = (1j * k0[:, None] * np.sqrt(np.clip(q_chunk[None, :]**2 - 1., 0.0, None)))

        R_chunk = _reflectivity_numba_core(h, vp, rho, omega_act, kx_chunk, 0., 0., False)
        # pass jacobian as vector of ones if weights already scaled
        _accumulate_evan_numba(k0, kx_chunk, kz_chunk, R_chunk, w_chunk, r_vec, dz_vec, dz_plus, dz_minus, hd, acc_evan, k0, free_surface=fs)

    acc_total = acc_prop + acc_evan 
    acc_total *= -1j / (2.0 * np.pi)
    # reconstruct full omega size with zeros for omega==0
    res_flat = np.zeros((Np, omega.size), dtype=np.complex128)
    res_flat[:, nz_mask] = acc_total
    # reshape to (Ns, Nr, Nw) if desired by caller, or return flattened (Np,Nw)
    res_pairs = res_flat.reshape((Ns, Nr, omega.size))
    return res_pairs   # shape (Ns, Nr, Nw)

# ---------- Numba accumulator for propagating part ----------
@njit(parallel=True, fastmath=True)
def _accumulate_prop_numba(k0, kx_chunk, kz_chunk, R_chunk, w_chunk, r_vec, dz_vec, dz_plus, dz_minus, h, out_acc, free_surface):
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
        dzp = dz_plus[p]
        dzm = dz_minus[p]
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
                term_direct = np.exp(1j * kz * dzm) * np.cos(kx * r)
                val = np.exp(1j * kz * dz) * np.cos(kx * r) * Rv + term_direct
                
                if free_surface:
                    Rfs = -1.0  # pressure release boundary
                    val = val + np.exp(1j * kz * dzp) * np.cos(kx * r) * Rfs
                    val = val / (1.0 - Rfs * Rv * np.exp(1j * 2.0 * kz * h))
                val -= term_direct
                acc += (val) * w_chunk[j]
            out_acc[p, iw] += acc #/ k0[iw]

# ---------- Numba accumulator for evanescent part ----------
@njit(parallel=True, fastmath=True)
def _accumulate_evan_numba(k0, kx_chunk, kz_chunk, R_chunk, w_chunk, r_vec, dz_vec, dz_plus, dz_minus, h, out_acc, jacobian_freq, free_surface):
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
        dzp = dz_plus[p]
        dzm = dz_minus[p]
        for iw in range(Nwa):
            acc = 0.0 + 0.0j
            kx_row = kx_chunk[iw]
            kz_row = kz_chunk[iw]
            R_row = R_chunk[iw]
            for j in range(chunk):
                kx = kx_row[j]
                kz = kz_row[j]
                Rv = R_row[j]

                term_direct = (1.0 / kz) * np.exp(1j * kz * dzm) * np.cos(kx * r)
                val = (1.0 / kz) * np.exp(1j * kz * dz) * np.cos(kx * r) * Rv + term_direct

                if free_surface:
                    Rfs = -1.0  # pressure release boundary
                    val = val + np.exp(1j * kz * dzp) * np.cos(kx * r) * Rfs * (1.0 / kz)
                    val = val / (1.0 - Rfs * Rv * np.exp(1j * 2.0 * kz * h))
                val -= term_direct
                acc += (val) * w_chunk[j]

               # val = (1.0 / kz) * np.exp(1j * kz * dz) * np.cos(kx * r) * Rv
               # acc += val * w_chunk[j]
                
            out_acc[p, iw] += acc * jacobian_freq[iw]
    
if __name__ == "__main__":
    #layers = [
    #    (100.0, 1500.0, 1800.0),
    #    (250.0, 1900.0, 2000.0),
    #    (350.0, 1700.0, 2200.0),
    #    (500.0, 2000.0, 2400.0),
    #]

    import time
    import matplotlib.pyplot as plt
    from utilities import green2d
    vp = 2000.
    layers = [(100.0, vp, 1000.0)]
    freqs = np.linspace(0.1, 100.0, 1024)
    omega = 2.0 * np.pi * freqs

    xs = 20.
    zs = 0.
    xr = 800.
    zr = 0.
    n_prop = 1024
    n_evan = 64

    start = time.time() # z_travel
    G_quad = integral_kx_quadrature_numba(
        layers, omega, xs, zs, xr, zr,
        n_prop, n_evan, kx_max_factor=2.0, chunk=256, fs=False)
    end = time.time()
    print(f"kx quadrature elapsed: {end-start:.2f} s")
    
    Green = green2d(omega, vp, xr-xs) 

    plt.figure()
    plt.plot(omega, np.real(Green), 'r--')
    plt.plot(omega, np.real(G_quad[0, 0, :]), 'b:')
    plt.xlabel('omega')
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(omega, np.imag(Green), 'g--')
    plt.plot(omega, np.imag(G_quad[0, 0, :]), 'b-.')
    plt.xlabel('omega')
    plt.grid()
    plt.show()

    '''start = time.time()
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
    print(f"kx quadrature elapsed: {end-start:.2f} s")'''