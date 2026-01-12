import sys
import os

# add src folder to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
from reflectivity_kx_omega import reflectivity
from layers import layers_to_arrays
#from src.quadrature.filon import composite_filon, precompute_quadrature_points
from quadrature.filon import precompute_quadrature_points, get_weights_filon
from acquisition import Acquisition
from utilities import green2d
from fortran.reflectivity_benchmark import fortran_reflectivity
from scipy.special import roots_legendre

def integrand_evan_cosh(psi, k0, z_abs, x):
    phase = z_abs *np.sinh(psi) - 1j * x *np.cosh(psi)
    return np.exp(-k0*phase)

def get_integrand_evan_param(kx_max_factor, nevan):
    psi_max = np.arccosh(kx_max_factor)
    # Gauss-Legendre quadrature points in [0, psi_max]
    pts, weights_leg = roots_legendre(nevan)
    scale_factor = psi_max / 2.0
    psi_i = scale_factor * (pts + 1.0)  # Map from [-1,1] to [0, psi_max]
    sinh_psi = np.sinh(psi_i[None, :])
    cosh_psi = np.cosh(psi_i[None, :])
    return sinh_psi, cosh_psi, psi_i, weights_leg, scale_factor

def build_node_map(subinterval_map, Nquad):
    """
    Global node -> list of (interval_id, local_node_id)
    """
    node_map = [[] for _ in range(Nquad)]
    for i, (start, end) in enumerate(subinterval_map):
        for j in range(end - start):
            node_map[start + j].append((i, j))
    return node_map

def Sommerfeld_integral(
    layers,
    omega,
    acq: Acquisition,
    Ntheta, Nevan=64,
    kx_max_factor=4.0, free_surface=False
):
    # layers
    h, vp, _ = layers_to_arrays(layers)
    # omegas
    Nw = omega.size
    k0_vec = omega / vp[0] 
    # ---- Acquisition geometry ----
    Ns, Nr = acq.xs.size, acq.xr.size

    dx_vec = np.abs(acq.xs[:, None] - acq.xr[None, :]).ravel()
    # dz_mat = np.abs(zs[:, None] - zr[None, :])
    dz_vec = np.abs(2*h[0] - acq.zr[:, None] - acq.zs[None, :]).ravel()
    Np = dx_vec.size

    # accumulators: (Np, Nwa)
    acc_prop = np.zeros((Np, Nw), dtype=np.complex128)
    acc_evan = np.zeros_like(acc_prop)

    thetas = np.linspace(-np.pi/2., np.pi/2., Ntheta)
    theta_eval, subinterval, Vinv = precompute_quadrature_points(thetas, 'chebychev')
    node_map = build_node_map(subinterval, len(theta_eval))

    start = time.time()
    p = np.sin(theta_eval) / vp[0]
    start = time.time()
    rmap = fortran_reflectivity(layers, omega, p, free_surface=1, zr=76., zs=76.)
    print('ntheta quadrature = ', theta_eval.shape)
    end = time.time()
    print(f"reflectivity prop elapsed: {end-start:.2f} s")

    sinh_psi, cosh_psi, psi_i, weights_ev, scaling = get_integrand_evan_param(kx_max_factor, Nevan)

    ph = np.cosh(psi_i) / vp[0]
    rmap_evan = fortran_reflectivity(layers, omega, ph, free_surface=1, zr=76., zs=76.)

    # Main loop over source-receiver pairs
    for p in range(Np):
        # get geometry
        dz = dz_vec[p]
        dx = dx_vec[p]
        # propagating part, integral of size (Np, Nw)
        weights_prop = get_weights_filon(k0_vec, dz, dx, thetas, theta_eval, Vinv, node_map) # shape (Nw, Nquad)
        acc_prop[p, :] = np.sum(weights_prop * rmap, axis=1) # Rmap of size # (Nw, Nquad)
        
        # evanescent part
        phase_min = dz * sinh_psi - 1j * dx * cosh_psi
        phase_plus = dz * sinh_psi + 1j * dx * cosh_psi
        F_i, F_i_neg = np.exp(-k0_vec[:, None] * phase_min), np.exp(-k0_vec[:, None] * phase_plus)

        integrand = (F_i + F_i_neg) * rmap_evan
        acc_evan[p, :] = scaling * np.sum(weights_ev[None, :] * integrand, axis=1)

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

