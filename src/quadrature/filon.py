import numpy as np
import numba as nb


''' 
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
'''

@nb.njit(fastmath=True)
def g(cos_t, sin_t, z_abs, x):
    """Phase function g(theta) = |z|*cos(theta) + x*sin(theta)"""
    return z_abs * cos_t + x * sin_t

@nb.njit(fastmath=True)
def g_prime(cos_t, sin_t, z_abs, x):
    """Derivative g'(theta) = -|z|*sin(theta) + x*cos(theta)"""
    return -z_abs * sin_t + x * cos_t

@nb.njit(fastmath=True)
def compute_filon_single(t, n):
    out = np.zeros(n, dtype=np.complex128)

    # small-theta closed forms if |t| < 0.25
    if abs(t) < 0.25:
        # up to 8 known values
        small = (2.0, 0.0, 2.0/3.0, 0.0, 2.0/5.0, 0.0, 2.0/7.0, 0.0)
        for k in range(n):
            out[k] = small[k]
        return out

    s = np.sin(t)
    c = np.cos(t)
    t2 = t * t
    t3 = t2 * t
    t4 = t2 * t2
    t5 = t4 * t
    t6 = t3 * t3

    if n > 0:
        out[0] = 2 * s / t
    if n > 1:
        out[1] = 2j * (-t * c + s) / t2
    if n > 2:
        out[2] = 2 * (t2 * s + 2 * t * c - 2 * s) / t3
    if n > 3:
        out[3] = 2j * (-t3 * c + 3 * t2 * s + 6 * t * c - 6 * s) / t4
    if n > 4:
        out[4] = 2 * (t4 * s + 4 * t3 * c - 12 * t2 * s - 24 * t * c + 24 * s) / t5
    if n > 5:
        out[5] = 2j * (-t5 * c + 5 * t4 * s + 20 * t3 * c - 60 * t2 * s - 120 * t * c + 120 * s) / t6
    return out

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
    Vinv, nodes = precompute_Vinv(order)
    n_nodes = nodes.size
    thetas = np.asarray(thetas)
    Nint = thetas.size - 1

    all_points = []
    global_idx = np.empty((Nint, n_nodes), dtype=np.int64)

    cur = 0
    for i in range(Nint):
        a, b = thetas[i], thetas[i+1]
        h = (b - a) / 2.0
        c = (a + b) / 2.0
        interval_points = c + h * nodes
        start_idx = cur
        # extend
        all_points.extend(interval_points.tolist())
        cur += n_nodes
        # fill global_idx for interval i
        for j in range(n_nodes):
            global_idx[i, j] = start_idx + j

    return np.asarray(all_points), Vinv, global_idx

@nb.njit(parallel=True, fastmath=True)
def get_weights_filon_numba(k0_vec, z, x, thetas, Vinv, global_idx, Weights):
    """
    Compute weights and directly assemble them to global quadrature nodes.

    Returns:
      Weights : array (Nw, Nquad) complex128
    """
    Nw = k0_vec.size
    Nint = thetas.size - 1
    n_nodes = Vinv.shape[0]

    # output: Nw x Nquad
    Weights[:] = 0.0 + 0.0j

    for i in nb.prange(Nint):
        a = thetas[i]
        b = thetas[i + 1]
        h = 0.5 * (b - a)
        c = 0.5 * (a + b)
        cos_c = np.cos(c)
        sin_c = np.sin(c)
        G = g(cos_c, sin_c, z, x)
        Gp = g_prime(cos_c, sin_c, z, x)

        # loop over frequencies
        for w in range(Nw):
            k0 = k0_vec[w]
            theta = Gp * k0 * h  # scalar
            phase = np.exp(1j * G * k0)
            tmp = compute_filon_single(theta, n_nodes) # shape (n_nodes,)
            for l in range(n_nodes):
                s = 0.0 + 0.0j
                for m in range(n_nodes):
                    s += tmp[m] * Vinv[l, m]
                gidx = global_idx[i, l]
                Weights[w, gidx] += s * h * phase
    return Weights

def get_weights_filon(k0_vec, z, x, thetas, Vinv, global_idx, weights):
    """
    Wrapper: ensures types and calls numba kernel.
    Inputs:
      Vinv: (n_nodes, n_nodes) complex128
      global_idx: (Nint, n_nodes) int64 mapping
    """
    # Ensure arrays are the right dtype
    k0_vec = np.ascontiguousarray(k0_vec, dtype=np.complex128)
    thetas = np.ascontiguousarray(thetas, dtype=np.float64)
    Vinv = np.ascontiguousarray(Vinv, dtype=np.float64)
    global_idx = np.ascontiguousarray(global_idx, dtype=np.int64)

    return get_weights_filon_numba(k0_vec, z, x, thetas, Vinv, global_idx, weights)
