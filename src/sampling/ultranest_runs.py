import numpy as np
import pickle
import ultranest

from src.config import Config
from src.layers import create_layers
from src.forward import ForwardSimulation
from src.noise import add_noise
from src.sampling.model import FWILogPosterior
from src.plot.plot_tools import plot_seismogram

if __name__ == "__main__":
    config = Config()

    vps_ref = np.array([1500.0, 1900.0, 2800.0, 3800.0, 2300.0, 5000.0])
    hs = np.array([150.0, 200.0, 250.0, 380.0, 440.0, 500.0])
    rhos = np.full_like(vps_ref, 2000.0)
    layers = create_layers(hs, vps_ref, rhos)

    sim = ForwardSimulation(config)
    d_clean, _ = sim.run(layers)
    d_obs, std_noise = add_noise(d_clean, config.noise_level, seed=config.seed)

    plot_seismogram(d_obs[0, :, :], sim.acq.xr, sim.param.time, vmin=-0.06, vmax=0.06)

    prior_mean = np.array([2000.0, 2500.0, 2500.0, 3000.0, 3000.0])
    sigma = np.array([1000.0, 1500.0, 1500.0, 1500.0, 1500.0])
    prior_cov = np.diag(sigma**2)

    model = FWILogPosterior(
        dobs=d_obs,
        layer=layers,
        config=config,
        noise_std=std_noise,
        prior_mean=prior_mean,
        prior_cov=prior_cov,
    )

    def ultranest_prior(cube):
        params = cube.copy()
        lo, hi = 1000, 7000
        for i in range(len(prior_mean)):
            params[i] = cube[i] * (hi - lo) + lo
        return params

    def ultranest_likelihood(params):
        c = np.asarray(params)
        return model(c)

    def ultranest_likelihood_weak(params):
        return ultranest_likelihood(params) / 16.0

    def ultranest_likelihood_strong(params):
        return 16.0 * ultranest_likelihood(params)

    vp0 = prior_mean.copy()
    print("log_prior(mu):", model.log_prior(vp0))
    print("log_likelihood(mu):", model.log_likelihood(vp0))

    vp_test = vp0 + sigma
    print("log_prior(mu + sigma):", model.log_prior(vp_test))
    print("log_likelihood(mu + sigma):", model.log_likelihood(vp_test))

    print("log_likelihood(ref):", model.log_likelihood(vps_ref[1:]))

    param_names = ["Vp^2", "Vp^3", "Vp^4", "Vp^5", "Vp^6"]
    sampler = ultranest.ReactiveNestedSampler(
        param_names, ultranest_likelihood, ultranest_prior
    )

    result = sampler.run(
        min_ess=500,
        min_num_live_points=1500,
        frac_remain=1e-2,
        max_ncalls=500000,
    )
    sampler.print_results()

    filename = "results_ultranest"
    with open(filename + ".pkl", "wb") as fp:
        pickle.dump(result, fp)
        print("results saved!")
