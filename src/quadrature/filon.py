import numpy as np
import os
import sys

import numba as nb

# Add src folder to Python path if running from an outer directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_FILON_CACHE = {}

def g(theta, z_abs, x):
    """Phase function g(theta) = |z|*cos(theta) + x*sin(theta)"""
    return z_abs * np.cos(theta) + x * np.sin(theta)

def g_prime(theta, z_abs, x):
    """Derivative g'(theta) = -|z|*sin(theta) + x*cos(theta)"""
    return -z_abs * np.sin(theta) + x * np.cos(theta)

'''def filon_quad_core(a, b, k0, z, x, Rvals, Vinv):
    """
    Compute Filon integral over [a,b] using:
    - order in {'quadratic','cubic','quartic'}
    """
    h = (b - a)/2
    c = (a + b)/2
    theta_bar = (k0 * g_prime(c, z, x)) * h
    n = len(Rvals)

    # quadratic closed form
    if n == 3:
        R_a, R_c, R_b = Rvals
        if np.isclose(theta_bar, 0.0):
            I = h*(R_a + 4*R_c + R_b)/3
        else:
            s, cth = np.sin(theta_bar), np.cos(theta_bar)
            tb = theta_bar
            alpha = (tb*s + cth - 1)/tb**2
            beta = (2*(1 - cth))/tb**2
            gamma = 2*(s - tb*cth)/tb**3
            I = h*(alpha*(R_a + R_b) + beta*R_c) + 1j*h*(gamma*(R_b - R_a))
    else:
        # Compute complex weights: W = Vinv @ Im
        Im = filon_moments(theta_bar, n)
        W = Vinv @ Im
        I = h * np.dot(W, Rvals)

    # Phase correction
    return I * np.exp(1j * k0 * g(c, z, x))

def composite_filon(thetas, k0, z, x, Rvals, subinterval_map, Vinv):
    total_integral = 0.0 + 0.0j # Initialize complex sum
    for i, (start_idx, end_idx) in enumerate(subinterval_map):
        a, b = thetas[i], thetas[i+1]
        # Extract R values for this subinterval
        R_subinterval = Rvals[start_idx:end_idx]
        # Filon integration on this subinterval
        subinterval_integral = filon_quad_core(
            a, b, k0, z, x, R_subinterval, Vinv
        )
        total_integral += subinterval_integral
    return total_integral'''

#   Closed-form oscillatory moments
@nb.njit(parallel=True, fastmath=True)
def filon_moments(theta, n):
    theta = np.asarray(theta)
    orig_shape = theta.shape
    flat = theta.reshape(-1)
    N = flat.size

    out = np.zeros((N, n), dtype=np.complex128)

    for i in nb.prange(N):
        t = flat[i]

        if abs(t) < 0.25:
            # Small-theta closed forms
            small = (2.0, 0.0, 2/3, 0.0, 2/5, 0.0, 2/7, 0.0)
            m = min(n, 8)
            for k in range(m):
                out[i, k] = small[k]
            continue

        s = np.sin(t)
        c = np.cos(t)

        t2 = t * t
        t3 = t2 * t
        t4 = t2 * t2
        t5 = t4 * t
        t6 = t3 * t3

        if n > 0:
            out[i, 0] = 2 * s / t
        if n > 1:
            out[i, 1] = 2j * (-t * c + s) / t2
        if n > 2:
            out[i, 2] = 2 * (t2 * s + 2 * t * c - 2 * s) / t3
        if n > 3:
            out[i, 3] = 2j * (-t3 * c + 3 * t2 * s + 6 * t * c - 6 * s) / t4
        if n > 4:
            out[i, 4] = (
                2 * (t4 * s + 4 * t3 * c - 12 * t2 * s
                     - 24 * t * c + 24 * s) / t5
            )
        if n > 5:
            out[i, 5] = (
                2j * (-t5 * c + 5 * t4 * s + 20 * t3 * c
                      - 60 * t2 * s - 120 * t * c + 120 * s) / t6
            )

    return out.reshape(orig_shape + (n,))

#   Precompute V^{-1} for an order (3,4,5 pts)
def precompute_Vinv(order):
    """Return the inverse Vandermonde matrix for standard nodes on [-1,1]."""
    if order == 'quadratic':    # 3 points
        nodes = np.array([-1, 0, 1], dtype=float)
    elif order == 'cubic':      # 4 points
        nodes = np.array([-1, -1/3, 1/3, 1], dtype=float)
    elif order == 'quartic':    # 5 points
        nodes = np.array([-1, -1/2, 0, 1/2, 1], dtype=float)
    elif order == 'chebychev':
        n = 6
        nodes = np.cos((2*np.arange(n)+1) * np.pi / (2*n))
    else:
        raise ValueError("order must be 'quadratic', 'cubic', or 'quartic'")

    n = len(nodes)
    V = np.vander(nodes, N=n, increasing=True).T
    return np.linalg.inv(V), nodes

def precompute_quadrature_points(thetas, order):
    if order not in _FILON_CACHE:
        Vinv, nodes = precompute_Vinv(order)
        _FILON_CACHE[order] = (Vinv, nodes)
    else:
        Vinv, nodes = _FILON_CACHE[order]

    n_intervals = len(thetas) - 1
    # Collect all evaluation points
    all_points = []
    subinterval_map = []
    
    for i in range(n_intervals):
        a, b = thetas[i], thetas[i+1]
        h = (b - a) / 2.0
        c = (a + b) / 2.0
        # Map nodes to physical coordinates
        interval_points = c + h * nodes
        start_idx = len(all_points)
        all_points.extend(interval_points)
        end_idx = len(all_points)
        subinterval_map.append((start_idx, end_idx))
    return np.array(all_points), subinterval_map, Vinv

'''def get_weights_filon(k0_vec, z, x, thetas, theta_eval,
                                 subinterval_map, Vinv):
    """
    Vectorized Filon weights for multiple frequencies.
    
    Args:
        k0_vec : array of shape (Nw,)
        z, x   : source position
        thetas : original partition
        theta_eval : full list of all quadrature nodes
        subinterval_map : list of (start,end)
        Vinv   : inverse Vandermonde (n_nodes, n_nodes)
    
    Returns:
        Weights : complex array of shape (Nw, Nquad)
    """
    k0_vec = np.asarray(k0_vec)
    Nw = len(k0_vec)
    Nquad = len(theta_eval)
    n_nodes = Vinv.shape[0]

    Weights = np.zeros((Nw, Nquad), dtype=np.complex128)

    for idx, (start, end) in enumerate(subinterval_map):
        a = thetas[idx]
        b = thetas[idx + 1]
        h = (b - a) / 2.0
        c = (a + b) / 2.0

        G  = g(c,  z, x)       # scalar
        Gp = g_prime(c, z, x)  # scalar

        # vector of theta
        theta_bar = (k0_vec * Gp) * h 
        phase     = np.exp(1j * k0_vec * G) # shape (Nw,)

        # number of nodes in the interval
        n = end - start
        if n != n_nodes:
            raise ValueError("Mismatch between Vinv and interval node count.")

        Im = filon_moments(theta_bar, n) 
        W  = Im @ Vinv.T 
        # Store weights
        Weights[:, start:end] = h * W * phase[:, None]

    return Weights'''

import time
def get_weights_filon(k0_vec, z, x, thetas, theta_eval, Vinv, node_map):
    """
    Vectorized Filon weights for multiple frequencies.

    Args:
        k0_vec : array of shape (Nw,)
        z, x   : source position
        thetas : original partition (Nint + 1,)
        theta_eval : full list of all quadrature nodes
        Vinv   : inverse Vandermonde (n_nodes, n_nodes)
        node_map : list of (interval_id, local_node_id) for each global node
    Returns:
        Weights : complex array of shape (Nw, Nquad)
    """
    Nw = k0_vec.size
    Nquad = len(theta_eval)
    n_nodes = Vinv.shape[0]
   
    a = thetas[:-1]
    b = thetas[1:]
    h = 0.5 * (b - a)
    c = 0.5 * (a + b)
    # Phase functions
    G  = g(c, z, x)
    Gp = g_prime(c, z, x)
    # theta_bar: (Nint, Nw)
    theta_bar = (Gp[:, None] * k0_vec[None, :]) * h[:, None]
    # phase: (Nint, Nw)
    phase = np.exp(1j * G[:, None] * k0_vec[None, :])
    # Im: (Nint, Nw, n_nodes)
    Im = filon_moments(theta_bar, n_nodes)
    # Apply inverse Vandermonde
    # W: (Nint, Nw, n_nodes)
    W = Im @ Vinv.T
    # Scaling
    W *= h[:, None, None]
    W *= phase[:, :, None]

    max_entries = max(len(l) for l in node_map)
    entries_i = -np.ones((Nquad, max_entries), dtype=np.int64)
    entries_j = -np.ones((Nquad, max_entries), dtype=np.int64)
    lengths = np.zeros(Nquad, dtype=np.int64)

    #start = time.time()
    for node, lst in enumerate(node_map):
        lengths[node] = len(lst)
        for k, (i, j) in enumerate(lst):
            entries_i[node, k] = i
            entries_j[node, k] = j
    # Scatter into global quadrature array
    Weights = assemble_weights_nodecentric(W, entries_i, entries_j, lengths, Nquad)
    return Weights

@nb.njit(parallel=True, fastmath=True)
def assemble_weights_nodecentric(W, entries_i, entries_j, lengths, Nquad):
    Nint, Nw, n_nodes = W.shape
    Weights = np.zeros((Nw, Nquad), dtype=np.complex128)

    for node in nb.prange(Nquad):
        for k in range(lengths[node]):
            i = entries_i[node, k]
            j = entries_j[node, k]
            for w in range(Nw):
                Weights[w, node] += W[i, w, j]
    return Weights
