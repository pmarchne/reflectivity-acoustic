import sys
import os

# add src folder to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
from reflectivity_kx_omega import reflectivity
from layers import layers_to_arrays
#from src.quadrature.filon import composite_filon, precompute_quadrature_points
from quadrature.filon import composite_filon, precompute_quadrature_points, get_weights_filon
from acquisition import Acquisition
from utilities import green2d
from scipy.special import roots_legendre

def integrand_evan_cosh(psi, k0, z_abs, x):
    phase = z_abs *np.sinh(psi) - 1j * x *np.cosh(psi)
    return np.exp(-k0*phase)

def Sommerfeld_integral(
    layers,
    omega,
    xs, zs, xr, zr,
    Ntheta, Nevan,
    kx_max_factor=4.0
):
    # layers
    h, vp, _ = layers_to_arrays(layers)
    # ---- Acquisition geometry ----
    xs = np.asarray(xs).ravel()
    zs = np.asarray(zs).ravel()
    xr = np.asarray(xr).ravel()
    zr = np.asarray(zr).ravel()
    Ns, Nr = xs.size, xr.size
    dx_mat = np.abs(xs[:, None] - xr[None, :])
    dz_mat = np.abs(2*h[0] - zr[:, None] - zs[None, :])
    dx_vec = dx_mat.ravel()
    dz_vec = dz_mat.ravel()
    Np = dx_vec.size

    # omegas
    omegas = np.atleast_1d(omega)
    Nw = omegas.size
    # accumulators: (Np, Nwa)
    acc_prop = np.zeros((Np, Nw), dtype=np.complex128)
    acc_evan = np.zeros_like(acc_prop)

    thetas = np.linspace(-np.pi/2., np.pi/2., Ntheta)
    order = 'quartic'
    theta_eval, subinterval, Vinv = precompute_quadrature_points(thetas, order)
    k0_vec = omegas / vp[0] 

    psi_max = np.arccosh(kx_max_factor)
    # Get Gauss-Legendre quadrature points in [0, psi_max]
    pts, weights_leg = roots_legendre(Nevan)
    scale_factor = psi_max / 2.0
    psi_i = scale_factor * (pts + 1.0)  # Map from [-1,1] to [0, psi_max]

    taper = np.hanning(len(theta_eval))
    ZR, ZS = 76., 76.
    Rmap = reflectivity(layers, omegas, theta_eval, ZR, ZS, mode="k0",  use_numba=True, fs=True) 
    Rmap *= taper[None, :]
    Rmap_evan = reflectivity(layers, omegas, psi_i, ZR, ZS, mode="psi",  use_numba=True, fs=True)
    taper_psi = np.hanning(len(psi_i)) 
    #Rmap = np.ones((Nw, len(theta_eval))) 
    #Rmap_evan = np.ones((Nw, len(psi_i))) 

    # ---- Main loop (to be parallelized over p and w) ----
    acc_prop = np.zeros((Np, Nw), dtype=np.complex128)
    acc_evan = np.zeros((Np, Nw), dtype=np.complex128)
    for p in range(Np):
        dz = dz_vec[p]
        dx = dx_vec[p]
        # acc prop of size (Np, Nw)
        Weights = get_weights_filon(k0_vec, dz, dx, thetas, theta_eval, subinterval, Vinv) # shape (Nw, Nquad)
        acc_prop[p,:] = np.sum(Weights * Rmap, axis=1) # Rmap of size # (Nw, Nquad)
        #acc_prop[p,:] = np.einsum('ij,ij->i', Weights, Rmap)
        #for w in range(len(omegas)):
        #    k0 = omega[w] / vp[0]
        #    R_w = Rmap[w, :]
        #    acc_prop[p, w] = composite_filon(thetas, k0, dz, dx, R_w, subinterval, Vinv)
        #for w in range(len(omegas)):
            #R_evan = Rmap_evan[w, :]
            #k0 = omega[w] / vp[0]
        # add the two parts of the integrals
        phase_min = dz *np.sinh(psi_i[None, :] ) - 1j * dx *np.cosh(psi_i[None, :] )
        phase_plus = dz * np.sinh(psi_i[None, :] ) + 1j * dx *np.cosh(psi_i[None, :])
        F_i = np.exp(-k0_vec[:, None] * phase_min)
        F_i_neg = np.exp(-k0_vec[:, None] * phase_plus)
        integrand = (F_i + F_i_neg) * Rmap_evan * taper_psi[None, :]
        acc_evan[p, :] = scale_factor * np.sum(weights_leg[None, :] * integrand, axis=1) 
        #acc_evan[p, :] = scale_factor * np.einsum('j,ij->i', weights_leg, integrand)

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

