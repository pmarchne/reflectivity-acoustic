import numpy as np
import numba as nb
from src.quadrature.gauss_lobatto import gauss_lobatto_nodes

""" 
Propagative regime : after the substitution kx = k0 sin(theta), the 2D Sommerfeld integral takes the form
            I_prop(x,z,w) = 1/(2*i)*int_{-pi/2}^{pi/2} R(theta, w) e^{i k_0 * g(theta)}
with 
    - k0 = w / vp_{top}
    - R(theta, w): the reflectivity map
    - g(theta) = |z|*cos(theta) + x*sin(theta)

Evanescent regime : after the substitution s = k0 cosh(psi), the 2D Sommerfeld integral takes the form
            I_evan(x,z,w) = -1/(4*pi)*int_{0}^{psi_max} R(psi, w) e^{-k_0 * h(psi)}
with 
    - k0 = w / vp_{top}
    - R(psi, w): the reflectivity map
    - h(psi) = |z|*sinh(psi) + i*x*cosh(psi)
"""


@nb.njit(fastmath=True)
def g(cos_t, sin_t, z_abs, x):
    """Phase function g(theta) = |z|*cos(theta) + x*sin(theta)"""
    return z_abs * cos_t + x * sin_t


@nb.njit(fastmath=True)
def g_prime(cos_t, sin_t, z_abs, x):
    """Derivative g'(theta) = -|z|*sin(theta) + x*cos(theta)"""
    return -z_abs * sin_t + x * cos_t


def nodes_and_endpoint_policy(kind="chebychev", n=6):
    if kind == "quadratic":
        nodes = np.array([-1.0, 0.0, 1.0], dtype=float)
        n = 3
        share_endpoints = True

    elif kind == "cubic":
        nodes = np.array([-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0], dtype=float)
        n = 4
        share_endpoints = True

    elif kind == "quartic":
        nodes = np.array([-1.0, -1.0 / 2.0, 0.0, 1.0 / 2.0, 1.0], dtype=float)
        n = 5
        share_endpoints = True

    elif kind == "chebychev":
        nodes = np.cos((2 * np.arange(n) + 1) * np.pi / (2 * n))
        share_endpoints = False

    elif kind == "gauss_lobatto":
        nodes = gauss_lobatto_nodes(n)
        share_endpoints = True
    else:
        raise ValueError("invalid rule")

    return nodes, share_endpoints


def build_filon_rule(thetas, kind="chebychev", n=6):
    """
    Build the small setup object used by the propagative kernel.

    Returns a dict with:
      nodes, share_endpoints, T, n_nodes, theta_eval
    """
    nodes, share = nodes_and_endpoint_policy(kind, n)
    n = nodes.size

    # nodal values -> coefficients in power basis on [-1,1]
    V = np.vander(nodes, N=n, increasing=True).T
    T = np.linalg.solve(V, np.eye(n))

    thetas = np.asarray(thetas, dtype=np.float64)
    Nint = thetas.size - 1
    theta_panel = np.empty((Nint, n), dtype=np.float64)

    for i in range(Nint):
        a, b = thetas[i], thetas[i + 1]
        h = 0.5 * (b - a)
        c = 0.5 * (a + b)
        vals = c + h * nodes
        # enforce endpoints
        if share:
            vals[0] = a
            vals[-1] = b
        theta_panel[i, :] = vals

    theta_flat = theta_panel.ravel()
    if share:
        theta_unique, inverse = np.unique(theta_flat, return_inverse=True)
    else:
        theta_unique = theta_flat
        inverse = np.arange(theta_flat.size)


    return {
        "kind": kind,
        "nodes": nodes,
        "share_endpoints": share,
        "T": T,
        "theta_eval": theta_unique,
        "inverse": inverse,
    }


@nb.njit(cache=True, inline='always')
def filon_moments(t, M):
    n_nodes = M.shape[0]

    # small-t series for stability
    if abs(t) < 0.25:
        K = 12
        for m in range(n_nodes):
            acc = 0.0 + 0.0j
            it_pow = 1.0 + 0.0j
            fact = 1.0
            for k in range(K + 1):
                power = m + k
                if (power & 1) == 0:
                    acc += it_pow * (2.0 / (power + 1.0)) / fact
                it_pow *= 1j * t
                fact *= (k + 1.0)
            M[m] = acc
        return

    et = np.exp(1j * t)
    em = np.exp(-1j * t)
    it = 1.0 / (1j * t)

    M[0] = (et - em) * it

    for m in range(1, n_nodes):
        sign = 1.0 if (m & 1) == 0 else -1.0
        boundary = et - sign * em
        M[m] = boundary * it - (m * it) * M[m - 1]


@nb.njit(parallel=True, fastmath=True, cache=True)
def transform_reflectivity(rmap, Nint, n_nodes, T, index_map):
    Nw = rmap.shape[0]
    R_trans = np.empty((Nint, Nw, n_nodes), dtype=np.complex128)
    for i in nb.prange(Nint):
        base = i * n_nodes
        for w in range(Nw):
            for m in range(n_nodes):
                s = 0.0 + 0.0j
                for l in range(n_nodes):
                    idx = index_map[base + l]
                    s += T[l, m] * rmap[w, idx]
                R_trans[i, w, m] = s
    return R_trans


@nb.njit(parallel=True, fastmath=True, cache=True)
def compute_prop_direct(dz_vec, dx_vec, k0_vec, thetas, R_trans):
    Np = dz_vec.size
    Nw = k0_vec.size
    Nint = thetas.size - 1
    n_nodes = R_trans.shape[2]

    acc_prop = np.zeros((Np, Nw), dtype=np.complex128)

    # precompute geometry
    mid_thetas = 0.5 * (thetas[:-1] + thetas[1:])
    h_vec = 0.5 * (thetas[1:] - thetas[:-1])
    cos_c = np.cos(mid_thetas)
    sin_c = np.sin(mid_thetas)

    for p in nb.prange(Np):
        dz = dz_vec[p]
        dx = dx_vec[p]
        res_p = np.zeros(Nw, dtype=np.complex128)
        M = np.empty(n_nodes, dtype=np.complex128) 
        for i in range(Nint):
            h = h_vec[i]
            G = dz * cos_c[i] + dx * sin_c[i]
            Gp = -dz * sin_c[i] + dx * cos_c[i]
            t_factor = Gp * h
            for w in range(Nw):
                k0 = k0_vec[w]
                t = t_factor * k0
                filon_moments(t, M)
                phase = np.exp(1j * G * k0)
                s = 0.0 + 0.0j
                coeff = R_trans[i, w]  # contiguous over m
                for m in range(n_nodes):
                    s += M[m] * coeff[m]
                res_p[w] += h * phase * s
        acc_prop[p, :] = res_p
    return acc_prop