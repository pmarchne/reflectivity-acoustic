import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
import pickle
from sampling.parametrization import diagnostic_checks
from src.utilities import timer
from fd_experiment import prepare_fd_model
from unconstrainedtarget import UnconstrainedTarget

try:
    from sampling_toolbox.svgd import SVGD
    from sampling_toolbox.gauss_vi import GaussianODE
    from sampling_toolbox.utilities.kl_tracker import RelativeKLTracker
    print("Success: Sampling toolbox loaded.")
except ImportError as e:
    print(f"Error: Could not find the toolbox. {e}")

import logging
# Configure root logger to output INFO messages to stdout/console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)


def initialize_mixture_and_particles(
    bayes,
    target,
    n_gaussians,
    nparticles,
    rng,
    use_unconstrained=True,
    std_init=200.0,
):
    """
    Initialize MixGVI Gaussian components and equivalent SVGD particles
    from the same initial mixture distribution.
    """
    dim = len(bayes.mu)
    # 1. Initialize Gaussian mixture in physical space
    mu_m = [
        np.clip(
            rng.normal(loc=3000.0, scale=500.0, size=dim),
            bayes.v_min + 1e-3,
            bayes.v_max - 1e-3,
        )
        for _ in range(n_gaussians)
    ]
    weights = np.ones(n_gaussians) / n_gaussians
    # 2. Convert mixture to optimization space
    if use_unconstrained:
        mu_in = [
            target.m_to_z(m)
            for m in mu_m
        ]
        R_in = []
        for m in mu_m:
            norm_m = (m - bayes.v_min) / target.range
            dm_dz = (
                target.range
                * norm_m
                * (1.0 - norm_m)
            )
            std_z = std_init / dm_dz
            R_in.append(
                np.diag(std_z)
            )
    else:
        mu_in = mu_m
        R_init = np.eye(dim) * std_init
        R_in = [
            R_init.copy()
            for _ in range(n_gaussians)
        ]
    # 3. Sample SVGD particles from same mixture
    component_ids = rng.choice(
        n_gaussians,
        size=nparticles,
        p=weights,
    )
    init_m = np.zeros((nparticles, dim))
    for i, k in enumerate(component_ids):
        z = (
            mu_in[k]
            +
            R_in[k] @ rng.normal(size=dim)
        )
        if use_unconstrained:
            init_m[i] = target.z_to_m(z)
        else:
            init_m[i] = z
    init_m = np.clip(
        init_m,
        bayes.v_min + 1e-3,
        bayes.v_max - 1e-3,
    )
    init_particles = (
        target.m_to_z(init_m)
        if use_unconstrained
        else init_m
    )
    return (mu_in, R_in, weights, init_particles, init_m)


def run_svgd(bayes, nparticles, nsteps, dt, seed, verbose=False, save=False, use_unconstrained=True, kernel_scale=1.0):
    target = UnconstrainedTarget(bayes) if use_unconstrained else bayes
    rng = np.random.default_rng(seed)

    print("initial dt = ", dt)
    svgd = SVGD(log_and_grad_post=target.log_and_grad_post,
                step_size=dt,
                n_iter=nsteps,
                kernel_scale=kernel_scale,
                rng=rng,
                tol=1e-5,
                verbose=True,
                transform=target.z_to_m)
    svgd.kl_track = RelativeKLTracker(141.86576429722984)
    print("\nRunning SVGD...")
    dim = len(bayes.mu)
    _, _, _, init_particles, init_m = initialize_mixture_and_particles(
        bayes,
        target,
        n_gaussians=10,
        nparticles=nparticles,
        rng=rng,
        use_unconstrained=use_unconstrained,
        std_init=200.0,
    )
    print("SVGD initial particles shape:", init_particles.shape)
    if verbose:
        print(f"init particles max: {np.max(init_m, axis=0)}")
        print(f"init particles min: {np.min(init_m, axis=0)}")
        print(f"Initial mean 0: {np.mean(init_m, axis=0)}")
        print(f"Initial std 0: {np.std(init_m, axis=0)}")
    with timer("svgd"):
        s_svgd, s_history, diagnostics = svgd.sample(init_particles)
    # Convert results back to physical space
    s_history = (
        [target.z_to_m(step) for step in s_history]
        if use_unconstrained
        else s_history
    )
    s_svgd = target.z_to_m(s_svgd) if use_unconstrained else s_svgd
    if verbose:
        svgd.report_calls()
        svgd.print_statistics(s_svgd)
        print(diagnostics)
    if save:
        filename = 'res_svgd_unconst_rng' + str(seed)
        with open(filename + '.pkl', 'wb') as fp:
            pickle.dump([s_history, diagnostics], fp)
            print("svgd results saved !")
    print("svgd finished")
    return s_history


def run_mix_gvi(
    bayes,
    nit,
    seed,
    n_gaussians=10,
    use_unconstrained=True,
    dt=0.001,
    integrator="heun_adaptive",
    save=False
):
    target = UnconstrainedTarget(bayes) if use_unconstrained else bayes
    rng = np.random.default_rng(seed)
    mu_in, R_in, _, _, _ = initialize_mixture_and_particles(
            bayes,
            target,
            n_gaussians=n_gaussians,
            nparticles=10,
            rng=rng,
            use_unconstrained=use_unconstrained,
            std_init=200.0,
        )
    print(
        "Initial mixture means (physical):\n",
        np.array([target.z_to_m(m) if use_unconstrained else m for m in mu_in]),
    )
    gvi = GaussianODE(
        target.log_and_grad_post,
        step_size=dt,
        n_iter=nit,
        time_scheme=integrator,
        precond="natural",
        step_size_w=0.1,
        time_scheme_fr=integrator,
    )
    gvi.kl_track = RelativeKLTracker(141.86576429722984)
    final_mean, final_R, final_ws, means, Rs, ws, kl_hist = gvi.sample(
        [m.copy() for m in mu_in], [r.copy() for r in R_in]
    )
    # Map trajectory back to physical space if unconstrained
    if use_unconstrained:
        means_m = [
            [target.z_to_m(comp_mean) for comp_mean in step_means]
            for step_means in means
        ]
        # Convert full Rs trajectory
        Rs_m = []
        for step_idx, step_means in enumerate(means):
            step_Rs_m = []
            for comp_idx, z_comp in enumerate(step_means):
                s = expit(z_comp)
                dm_dz = target.range * s * (1.0 - s)
                R_z = Rs[step_idx][comp_idx]
                cov_z = R_z @ R_z.T
                cov_m_comp = np.diag(dm_dz) @ cov_z @ np.diag(dm_dz)
                step_Rs_m.append(np.linalg.cholesky(cov_m_comp))
            Rs_m.append(step_Rs_m)
        std = np.sqrt(np.diag(Rs_m[-1][0] @ Rs_m[-1][0].T))
    else:
        means_m = means
        Rs_m = Rs
        cov_m = final_R[0] @ final_R[0].T
        std = np.sqrt(np.diag(cov_m))

    gvi.report_calls()
    print("GVI finished")
    filename = f"test_unconst_MoG{n_gaussians}_rng{seed}"
    if save:
        with open(filename + ".pkl", "wb") as fp:
            pickle.dump([means_m, Rs_m, ws, kl_hist, gvi.dt_history], fp)
            print("Results saved!")
    return means_m, Rs_m, ws, kl_hist


def run_gvi(bayes, nit, mean_in=2500., name='2500', save=False):
    # Setup
    print("Starting GVI \n")
    dim = 7
    mu_in = [np.array([mean_in] * dim)] 
    std = 500.0
    cov_in = np.diag(np.full(dim, std**2))
    R_in = [np.linalg.cholesky(cov_in)]
    use_unconstrained = True
    target = UnconstrainedTarget(bayes) if use_unconstrained else bayes
    if use_unconstrained:
        mu_in, R_in = target.gmm_physical_to_unconstrained(mu_in, std)

    dt = 0.001
    integrator = 'heun_adaptive'
    gvi = GaussianODE(target.log_and_grad_post, step_size=dt, n_iter=nit, time_scheme=integrator, precond='natural', step_size_w=0., time_scheme_fr=integrator)
    gvi.kl_track = RelativeKLTracker(141.86576429722984)
    final_mean, final_R, final_ws, means, Rs, ws, kl_hist = gvi.sample([m.copy() for m in mu_in], [r.copy() for r in R_in])
    if use_unconstrained:
        means, Rs = target.gmm_unconstrained_to_physical(means, Rs)

    gvi.report_calls()
    print("gvi finished")
    filename = 'res_gvi_std500_mu'+name
    if save:
        with open(filename + '.pkl', 'wb') as fp:
            pickle.dump([means, Rs, ws, kl_hist, gvi.dt_history], fp)
            print("results saved !")
    return means, Rs, ws, kl_hist


def run_fwi_map(bayes, param):
    unique_map_minima = []
    num_starts = 1
    post_map = np.array([2696.8198354246188, 3074.4423873792193, 1980.826105977049, 3202.2884114407298, 3700.4895477382443, 2397.930062650532, 4124.598347538598])
    print("posterior MAP = ", -bayes.log_and_grad_post(post_map)[0])

    model_start = [
             np.array([2200.0] * 7),
             np.array([2500.0] * 7),
             np.array([2800.0] * 7),
             np.array([3000.0] * 7),
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



if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    seed = 42
    rng = np.random.default_rng(seed)
    m_ref = np.array([2700.0, 3200.0, 1900.0, 4200.0, 3800.0, 2200.0, 4500.0])
    obs_path = "FD_comparison/data/seis_v3_nofs"
    bayes, param = prepare_fd_model(file_path=obs_path, seed=seed, debug=False)
    param.vp_bounds = (1000.0, 6000.0) # 600 or 1000 !
    log_Z = 141.86576429722984
    diagnostic_checks(bayes, param.prior_transform, m_ref)
    run_fwi_map(bayes, param)
    nit = 200

    seed_init = 4 # or loop over seeds
    #run_gvi(bayes, nit, mean_in=2500., name='2500', save=False)
    run_gvi(bayes, nit, mean_in=2800., name='2800', save = False)
    #run_mix_gvi(bayes, nit=nit, seed=seed_init, n_gaussians=10, use_unconstrained=True, save=False)
    run_mix_gvi(bayes, nit=nit, seed=seed_init, n_gaussians=5, use_unconstrained=True, save=False)
    run_svgd(bayes, nparticles=200, nsteps=nit, dt=0.01, seed=seed_init, save=True, verbose=True, use_unconstrained=True, kernel_scale=1.)

    # cov_inv, hess = run_fwi_map_gaussian_approx(bayes, eps=3e-2)



