import numpy as np
from scipy.optimize import minimize

def run_fwi_map(bayes, param):
    unique_map_minima = []
    num_starts = 12
    post_map = np.array([2696.8198354246188, 3074.4423873792193, 1980.826105977049, 3202.2884114407298, 3700.4895477382443, 2397.930062650532, 4124.598347538598])
    print("posterior MAP = ", bayes.log_and_grad_post(post_map)[0])

    model_start = [
             np.array([2200.0] * 7),
             np.array([2500.0] * 7),
             np.array([2800.0] * 7),
             np.array([3000.0] * 7),
             np.array([2700.0, 3200.0, 1900.0, 4200.0, 3800.0, 4500.0, 2200.0]),
             np.array([2600., 2900., 3200., 3500., 3800., 4100., 4400.]),
             np.array([4000., 3700., 3400., 3100., 2800., 2500., 2200.]),
             np.array([2500., 3000., 2500., 2000., 2500., 3000., 2500.]),
             np.array([2500., 2000., 2500., 3000., 2500., 2000., 2500.]),
             np.array([3000., 3300., 3600., 3300., 3000., 2700., 2400.]),
             np.array([3000., 2700., 2400., 2700., 3000., 3300., 3600.]),
             np.array([2700., 3100., 1980., 3200., 3700., 4120., 2398.])
    ]

    all_bounds = [param.vp_bounds] * param.n_vp
    for i in range(num_starts):
        # Generate a random initial guess
        random_m_init = model_start[i]#param.draw_from_prior()
        print("Initial model :", random_m_init)
        print(f"--- Start {i+1}/{num_starts} ---")
        cost_history = []
        obj_func = make_map_objective(bayes, cost_history)
        res = minimize(
            obj_func,
            random_m_init,
            method="L-BFGS-B",
            jac=True,
            bounds=all_bounds,
            options={'maxiter': 50, 'ftol': 1e-6, 'gtol': 1e-4, 'disp': True}
        )
        print(res)
        print(f"{'-'*50}\nInverted Vp: {res.x}")
        # compute model L2 error and distance to the true posterior MAP
        rel_err = np.linalg.norm(res.x - post_map) / np.linalg.norm(post_map)
        print(f"{'-'*50}\n Rel L2 norm to map: {rel_err}")
        print(f"Completed in {res.nfev} evaluations.")
        
        if res.success:
            unique_map_minima.append({'vp': res.x, 'map_val': -res.fun})
            print(f"minimum found!")
        else:
            unique_map_minima.append({'vp': res.x, 'map_val': -res.fun})
            print(f"did not converged")
            
    print("\n=== All Discovered MAP Local Minima ===")
    # Sort them by objective value so the best candidate is at the top
    unique_map_minima = sorted(unique_map_minima, key=lambda x: x['map_val'])
    for idx, minimum in enumerate(unique_map_minima):
        print(f"MAP Mode #{idx}: Objective = {minimum['map_val']:.4e} | Vp = {minimum['vp'].astype(int)}")
    return 1

def make_map_objective(bayes, cost_history):
    """Define Regularized FWI objective function for MAP estimation"""
    def map_objective(model):
        #with timer("log and grad eval"):
        phi_map, grad_phi_map = bayes.log_and_grad_post(model)
        l2_misfit = bayes.l2_misfit(model)
        cost_history.append(phi_map)
        m_str = "[" + ", ".join(f"{m:7.1f}" for m in model) + "]"
        print(f"Iter {len(cost_history):>2} | MAP: {phi_map:.4e} (misift: {l2_misfit:.2e} | m: {m_str}")
        # because we maximize the MAP, so minimize -MAP
        return -phi_map, -grad_phi_map 
    return map_objective


def run_fwi_map_gaussian_approx(bayes, eps=1e-3):
    # estimated map after l-bfgs
    map_est = np.array([2700.692, 3062.192, 2014.517, 3081.276, 3697.679, 2174.89,  3283.172])
    print("\nComputing finite-difference inverse Hessian at the MAP estimate...")
    n_params = len(map_est)
    hessian = np.zeros((n_params, n_params))
    
    for i in range(n_params):
        m_forward = map_est.copy()
        m_back = map_est.copy()
        h = max(1.0, abs(map_est[i])) * eps
        print("step =", h)
        m_forward[i] += h
        m_back[i] -= h
        # Evaluate gradients at perturbed points
        _, grad_forward = bayes.log_and_grad_post(m_forward)
        _, grad_back = bayes.log_and_grad_post(m_back)
        neg_grad_forward = -grad_forward
        neg_grad_backward = -grad_back
        hessian[:, i] = (neg_grad_forward - neg_grad_backward) / (2.0 * h)
    # Symmetrize the Hessian matrix for numerical stability
    hessian = 0.5 * (hessian + hessian.T)
    try:
        covariance_matrix = np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        print("Warning: Hessian matrix is singular or ill-conditioned. Using pseudo-inverse instead.")
        covariance_matrix = np.linalg.pinv(hessian)

    print("\n=== Gaussian Approximation Results ===")
    print("Mean (MAP estimate):\n", map_est)
    return covariance_matrix, hessian