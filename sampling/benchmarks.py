import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pickle
from src.utilities import timer
from fd_experiment import prepare_fd_model
from parametrization import diagnostic_checks


try:
    from sampling_toolbox.svgd import SVGD
    from sampling_toolbox.gauss_vi import GaussianODE
    from sampling_toolbox.utilities.kl_tracker import GenericKLTracker
    print("Success: Sampling toolbox loaded.")
except ImportError as e:
    print(f"Error: Could not find the toolbox. {e}")


def load_reference():
    # load reference results from ultranest
    path = 'results_sampling/'
    with open(path + 'results_dahu_ultranest_nofs.pkl', 'rb') as fp:
        res = pickle.load(fp)
        print("results loaded !")
    samples_ref = res['samples']
    print("max likelihood :", res['maximum_likelihood']['point'])
    return samples_ref

def run_svgd(bayes, nparticles, nsteps, dt, rng, verbose=False, save=False):
    svgd = SVGD(log_likelihood=bayes.log_likelihood,
                log_prior=bayes.log_prior,
                grad_log_likelihood=bayes.grad_log_likelihood,
                grad_log_prior=bayes.grad_log_prior,
                step_size=dt,
                n_iter=nsteps,
                rng=rng,
                tol=1e-5)
    print("\nRunning SVGD...")

    mu = bayes.mu
    sigma = bayes.cov / 3.
    initial_particles = rng.normal(loc=mu, scale=np.sqrt(np.diag(sigma)), size=(nparticles, len(mu)))

    if verbose:
        print(f"init particles max: {np.max(initial_particles, axis=0)}")
        print(f"init particles min: {np.min(initial_particles, axis=0)}")
        print(f"Initial mean 0: {np.mean(initial_particles, axis=0)}")
        print(f"Initial std 0: {np.std(initial_particles, axis=0)}")

    with timer("svgd"):
        s_svgd, s_history, kl_hist_svgd = svgd.sample(initial_particles, num_samples=0)

    if verbose:
        svgd.report_calls()
        svgd.print_statistics(s_svgd)
    print(kl_hist_svgd)
    if save:
        filename = 'results_svgd'
        with open(filename + '.pkl', 'wb') as fp:
            pickle.dump([s_history, kl_hist_svgd], fp)
            print("svgd results saved !")
    print("svgd finished")
    return s_history


def run_mix_gvi(bayes, nit):
    # Setup
    mu_in = [np.array([2500.0, 2500.0, 2500.0, 2500.0, 2500.0, 2500.0, 2500.0]), np.array([3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0]), np.array([2000.0, 2800.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0]), np.array([3000.0, 3000.0, 2000.0, 3500.0, 3500.0, 2000.0, 3000.0])]#bayes.mu [bayes.mu]
    R_in = [np.linalg.cholesky(bayes.cov/8.), np.linalg.cholesky(bayes.cov/8.), np.linalg.cholesky(bayes.cov/8.), np.linalg.cholesky(bayes.cov/8.)]
    dt = 0.00001
    integrator = 'heun_adaptive'
    gvi = GaussianODE(bayes.log_and_grad_post, step_size=dt, n_iter=nit, time_scheme=integrator, precond='natural', step_size_w=0.1, time_scheme_fr=integrator)
    gvi.kl_track = GenericKLTracker(0.)

    final_mean, final_R, final_ws, means, Rs, ws, kl_hist = gvi.sample([m.copy() for m in mu_in], [r.copy() for r in R_in])
    
    final_cov = final_R[0] @ final_R[0].T
    std = np.sqrt(np.diag(final_cov))
    print(final_cov)
    print(final_ws)
    gvi.report_calls()
    print("gvi finished")
    print(std)
    filename = 'results_4gvi_mix'
    with open(filename + '.pkl', 'wb') as fp:
        pickle.dump([means, Rs, ws, kl_hist], fp)
        print("results saved !")
    print(final_mean)

    plt.figure(figsize=(6, 2.25))
    plt.plot(gvi.dt_history, color='g', label='w', lw=1.7, linestyle='-')
    plt.xlabel(r'Iteration')
    plt.xlim([0, 200])
    plt.ylabel(r'$\Delta t$')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(6, 2.25))
    plt.plot(kl_hist, color='g', label='nat', lw=1.7, linestyle='-')
    plt.xlabel(r'Iteration')
    plt.ylabel(r'KL$(\mu \parallel \pi)$')
    plt.grid(True)
    plt.legend()
    plt.show()

    return means, Rs, ws, kl_hist

def run_gvi(bayes, nit):
    # Setup
    mu_in = np.array([2500.0, 2500.0, 2500.0, 2500.0, 2500.0, 2500.0, 2500.0])#bayes.mu
    R_in = np.linalg.cholesky(bayes.cov/2.)
    dt = 0.0001
    integrator = 'heun_adaptive'
    gvi = GaussianODE(bayes.log_and_grad_post, step_size=dt, n_iter=nit, time_scheme=integrator, precond='natural', step_size_w=0., time_scheme_fr=integrator)
    gvi.kl_track = GenericKLTracker(5.0)

    final_mean_id, final_R_id, final_ws_id, means, Rs, ws, kl_hist = gvi.sample([m.copy() for m in mu_in], [r.copy() for r in R_in])
    
    #final_cov = final_R @ final_R.T
    #std = np.sqrt(np.diag(final_cov))
    gvi.report_calls()
    print("gvi finished")

    filename = 'results_gvi'
    with open(filename + '.pkl', 'wb') as fp:
        pickle.dump([means, Rs], fp)
        print("results saved !")
    
    return means, Rs, ws, kl_hist


def run_fwi_map(bayes, param):
    unique_map_minima = []
    num_starts = 3
    model_start = [np.array([3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0]), np.array([2800.0, 3400.0, 2400.0, 3000.0, 3000.0, 3000.0, 3000.0]), np.array([2500.0, 2500.0, 2500.0, 2500.0, 2500.0, 2500.0, 2500.0])]

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
            options={"maxiter": 50, "ftol": 1e-6, "gtol": 1e-4}
        )
        print(f"{'-'*50}\nInverted Vp: {res.x}")
        print(f"Completed in {len(cost_history)} evaluations.")
        
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
        with timer("log and grad eval"):
            phi_map, grad_phi_map = bayes.log_and_grad_post(model)
        l2_misfit = bayes.l2_misfit(model)
        cost_history.append(phi_map)
        m_str = "[" + ", ".join(f"{m:7.1f}" for m in model) + "]"
        print(f"Iter {len(cost_history):>2} | MAP: {phi_map:.4e} (misift: {l2_misfit:.2e} | m: {m_str}")
        # because we maximize the MAP, so minimize -MAP
        return -phi_map, -grad_phi_map 
    return map_objective

if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True)
    seed = 42
    rng = np.random.default_rng(seed)
    m_ref = np.array([2700.0, 3200.0, 1900.0, 4200.0, 3800.0, 2200.0, 4500.0])
    obs_path = "FD_comparison/data/seis_v3_nofs"
    bayes, param = prepare_fd_model(file_path=obs_path, seed=seed, debug=False)
    diagnostic_checks(bayes, param.prior_transform, m_ref)

    #run_fwi_map(bayes, param)
    #run_mix_gvi(bayes, nit=150)
    run_svgd(bayes, nparticles=100, nsteps=200, dt=50., rng=rng, save=True, verbose=True)
    #run_mix_gvi(bayes, dt=50, nit=100)

