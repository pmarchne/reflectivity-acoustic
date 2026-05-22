import os
import pickle
import numpy as np
import ultranest
import ultranest.stepsampler
import h5py
# from ultranest.plot import cornerplot
from scipy.stats import truncnorm
# from synthetic_experiment import prepare_synthetic_model
from fd_data_experiment import prepare_fd_model
from src.utilities import timer


def run_diagnostic_checks(bayes, prior_transform, v_true=None):
    """
    Performs sanity checks on the Likelihood, Prior, and Parameter Mapping.
    """
    mu = bayes.mu
    sigma = np.sqrt(np.diag(bayes.cov))

    print("\n" + "=" * 50)
    print(f"{'BAYESIAN FWI DIAGNOSTIC CHECKS':^50}")
    print("=" * 50)

    # 1. Check at Prior Mean
    with timer("prior eval"):
        lp_mu = bayes.log_prior(mu)
    with timer("likelihood eval"):
        ll_mu = bayes.log_likelihood(mu)
    print(f"[*] Prior Mean (mu): {mu}")
    print(f"    > log_prior:      {lp_mu:.4f}")
    print(f"    > log_likelihood: {ll_mu:.4f}")

    # 2. Check at mu + sigma (Sensitivity Check)
    vp_test = mu + sigma
    with timer("likelihood eval"):
        ll_sigma = bayes.log_likelihood(vp_test)
    delta_ll = ll_sigma - ll_mu
    print(f"[*] Shifted (mu + sigma): {vp_test}")
    print(f"    > log_likelihood: {ll_sigma:.4f}")
    print(f"    > Delta LL:       {delta_ll:.4f}")

    # 3. Reference Check (The 'Target' Score)
    if v_true is not None:
        with timer("likelihood eval"):
            ll_ref = bayes.log_likelihood(v_true)
        print(f"[*] True Reference:      {v_true}")
        print(f"    > log_likelihood: {ll_ref:.4f}")
        print(f"    > Distance to Target: {ll_ref - ll_mu:.4f} (log-units)")

    # 4. Range & Sensitivity Validation
    if np.isclose(ll_mu, ll_sigma, rtol=1e-5):
        print("\n!! WARNING: Likelihood is FLAT. The data is not sensitive to Vp.")
        print("   Check if noise_std is too large or if the simulation is running.")

    if np.abs(delta_ll) > 1e6:
        print("\n!! WARNING: Likelihood is extremely SHARP.")
        print(
            "   This might lead to poor sampling. Consider increasing noise_std or beta."
        )

    # 5. Prior Transform Mapping (Unit Cube -> Physical)
    test_cube = np.array([0.5] * len(mu))
    transformed = prior_transform(test_cube)
    print("\n[*] Prior Mapping Check:")
    print(f"    > Unit Cube [0.5] maps to: {transformed}")

    # Check if mapping stays in bounds
    if np.any(transformed < 1000) or np.any(transformed > 7000):
        print("    !! ERROR: Prior transform is mapping outside [1000, 7000]!")
    else:
        print("    > Range check: PASS")

    print("=" * 50 + "\n")


def run_ultranest_inference(bayes, v_true, prior_mode='gaussian'):

    def prior_transform(cube):
        """
        Maps UltraNest's [0, 1] unit cube to physical parameter space.
        Options:
        - mode='gaussian': Truncated Gaussian
        - mode='uniform': Uniform [1000, 7000]
        """
        # Bounds for Vp
        v_min, v_max = 1000.0, 7000.0
        if prior_mode == 'uniform':
            # Simple linear mapping: [0, 1] -> [min, max]
            return cube * (v_max - v_min) + v_min
        elif prior_mode == 'gaussian':
            mu = bayes.mu
            sigma = np.sqrt(np.diag(bayes.cov))
            a_scaled = (v_min - mu) / sigma
            b_scaled = (v_max - mu) / sigma
            # Map the entire cube in one shot
            return truncnorm.ppf(cube, a_scaled, b_scaled, loc=mu, scale=sigma)

    def log_likelihood(params):
        return bayes.log_likelihood(params)

    run_diagnostic_checks(bayes, prior_transform, v_true)

    param_names = [f"Vp_{i+1}" for i in range(len(v_true))]

    save_dir = "ultranest_FD_TOYxDAC"
    sampler = ultranest.ReactiveNestedSampler(
        param_names,
        log_likelihood,
        prior_transform,
        log_dir = save_dir,
        resume = "overwrite",
    )

    sampler.stepsampler = ultranest.stepsampler.SliceSampler(
        nsteps             = 4 * len(v_true),
        generate_direction = ultranest.stepsampler.generate_mixture_random_direction,
    )

    result = sampler.run(min_num_live_points=800, dlogz=0.5, min_ess=2000, update_interval_volume_fraction=0.4, max_num_improvement_loops=5)  # 1000
    sampler.print_results()

    print(f"\nEqual-weighted posterior samples: {samples.shape}")
    print(f"Effective sample size (ESS): {result['ess']:.0f}")
    print(f"log Z = {result['logz']:.3f} ± {result['logzerr']:.3f}")

    filename = save_dir+'results_ultranest'
    with open(filename + '.pkl', 'wb') as fp:
        pickle.dump(result, fp)
        print("results saved !")
    
    sampler.plot_run()
    sampler.plot_trace()
    sampler.plot_corner() 


if __name__ == "__main__":
    seed = 42
    np.random.seed(seed)
    # Link to the experiment file
    # v_ref = np.array([3500.0, 1800.0, 2300.0, 4000.0])
    #bayes_model = prepare_synthetic_model()
    v_ref = np.array([1603.0, 1749.0, 2019.0, 2179.0, 1900.0, 2265.0, 3281.0])
    path = "FD_comparison/data/seis_v1_nofs"
    bayes_model = prepare_fd_model(file_path=path, seed=seed)
    print("n cpu : ", os.cpu_count())
    # Launch UltraNest
    run_ultranest_inference(bayes_model, v_true=v_ref, prior_mode='gaussian')
