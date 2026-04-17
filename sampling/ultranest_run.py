import os
import numpy as np

# import matplotlib.pyplot as plt
import ultranest
import pickle
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


def run_ultranest_inference(bayes, v_true):
    # Bounds for Vp
    v_min, v_max = 1000.0, 7000.0

    mu = bayes.mu
    sigma = np.sqrt(np.diag(bayes.cov))

    def prior_transform(cube):
        """
        Maps UltraNest's [0, 1] space to a Truncated Gaussian [1000, 7000]
        """
        params = np.empty_like(cube)
        a_scaled = (v_min - mu) / sigma
        b_scaled = (v_max - mu) / sigma

        # Map the entire cube in one shot
        params = truncnorm.ppf(cube, a_scaled, b_scaled, loc=mu, scale=sigma)
        return params

    def log_likelihood(params):
        return bayes.log_likelihood(params)

    run_diagnostic_checks(bayes, prior_transform, v_true)

    param_names = [f"Vp_{i+1}" for i in range(len(mu))]

    sampler = ultranest.ReactiveNestedSampler(
        param_names,
        log_likelihood,
        prior_transform,
    )

    result = sampler.run(min_ess=50, min_num_live_points=300)
    sampler.print_results()

    filename = 'results_ultranest_nofs'
    with open(filename + '.pkl', 'wb') as fp:
        pickle.dump(result, fp)
        print("results saved !")
    # cornerplot(result, plot_density=True, title_kwargs={"fontsize": 16})
    # plt.show()

    # sampler.plot_corner()
    # plt.show()


if __name__ == "__main__":
    # Link to the experiment file
    # v_true = np.array([3000.0, 1800.0, 3500.0])
    # bayes_model = prepare_synthetic_model()
    v_ref = np.array([1603.0, 1749.0, 2019.0, 2179.0, 1900.0, 2265.0, 3281.0])
    bayes_model = prepare_fd_model()
    print("n cpu : ", os.cpu_count())
    # Launch UltraNest
    run_ultranest_inference(bayes_model, v_true=v_ref)
