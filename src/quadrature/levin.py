@nb.njit(parallel=True, fastmath=True, cache=True)
def compute_prop_levin(dz_vec, dx_vec, k0_vec, thetas, R_trans, D_ref):
    Np = dz_vec.size
    Nw = k0_vec.size
    Nint = thetas.size - 1
    n_nodes = R_trans.shape[1]

    mid_thetas = 0.5 * (thetas[:-1] + thetas[1:])
    h_vec = 0.5 * (thetas[1:] - thetas[:-1])
    cos_c = np.cos(mid_thetas)
    sin_c = np.sin(mid_thetas)

    acc_prop = np.zeros((Np, Nw), dtype=np.complex128)

    for p in nb.prange(Np):
        dz, dx = dz_vec[p], dx_vec[p]
        res_p = np.zeros(Nw, dtype=np.complex128)

        for i in range(Nint):
            h = h_vec[i]

            # local geometry
            # on the reference nodes, build g'(theta_j)
            # then map D_ref to the physical interval
            for w in range(Nw):
                k0 = k0_vec[w]

                # build local Levin system
                # A = D_phys + 1j * k0 * diag(gprime_nodes)
                # rhs = local reflectivity samples

                # solve for auxiliary coefficients
                # u = solve_small_system(A, rhs)

                # extract interval contribution
                # res_p[w] += interval_contribution(u, endpoints, phase)

        acc_prop[p, :] = res_p

    return acc_prop

def build_rule(order):
    nodes, share = nodes_and_endpoint_policy(order)
    n = len(nodes)

    V = np.vander(nodes, N=n, increasing=True).T
    T = np.linalg.solve(V, np.eye(n))

    D_ref = differentiation_matrix(nodes)   # needed for Levin

    return {
        "nodes": nodes,
        "share_endpoints": share,
        "n_nodes": n,
        "T": T,
        "D_ref": D_ref,
    }

''' def solve_sommerfeld(..., rule="chebychev_6", method="filon"):
    quad = build_rule(rule)          # nodes, T, maybe D
    theta_eval = build_theta_eval(thetas, quad["nodes"])
    R_prop = sample_reflectivity(theta_eval, ...)
    
    if method == "filon":
        return integrate_filon(...)
    elif method == "levin":
        return integrate_levin(...)
    else:
        raise ValueError("method must be 'filon' or 'levin'")
    
solve(rule="gauss_lobatto_5", method="filon")
solve(rule="chebychev_6", method="levin") '''