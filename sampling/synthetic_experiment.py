import numpy as np
from src.config import Config
from src.simulation import Simulation
from src.noise import add_noise_snr
from sampling.parametrization import ModelParameterization
from src.layers import create_layers
from src.utilities import estimate_neff
from sampling.posterior import FWIPosterior
from src.plot.plot_tools import plot_seismogram

def prepare_synthetic_model(seed):
    config = Config(
        n_receivers=8,
        x_min=600.0,
        x_max=2000.0,
        z_rec=50.0,
        z_src=20.0,
        x_src=50.0,
        nq_prop=256,
        nq_evan=128,
        f0=5.0,
        total_time=1.5,
        delay=0.2,
        epsilon=1.25,
        free_surface=False,
        nfft_pad_factor=1,
        source_deriv=False,
        ind_traces=[],
    )

    vp_ref = np.array([1500.0, 1800.0, 3500.0])
    h_ref = np.array([100.0, 150., 50.])
    rho = np.array([2000.0, 2000.0, 2000.0])

    layers = create_layers(h_ref, vp_ref, rho)
    sim = Simulation(config)

    # 2. Generate Synthetic "Observed" Data
    d_clean = sim.forward(layers)
    d_obs, std_noise = add_noise_snr(d_clean.squeeze(), snr_db=20, seed=seed)
    print(f"Estimated Noise Std: {std_noise:.4f}")

    # Prior Parameters
    mu_prior = np.array([2500.0, 2500.0, 200.0])
    cov_prior = np.diag([500**2, 500**2, 100**2])

    # Calculate effective samples
    factor = 1. # model mismatch inflation factor
    beta = factor * config.f0 * config.total_time / sim.param.nt
    # beta = 1. / d_clean.size
    # beta = factor*config.total_time * config.f0 / sim.param.nt
    print(f"Tempering factor beta: {beta:.4f}")

    param = ModelParameterization(
        layers,
        invert_vp=True,
        invert_h=True,
        startv=1,
        starth=1,
        prior_mode='gaussian',
        mu=mu_prior,
        cov=cov_prior
    )

    # Initialize the posterior object
    bayes = FWIPosterior(
        d_obs,
        param,
        sim,
        mu_prior,
        cov_prior,
        std_noise=float(std_noise),
        beta = beta,
    )

    return bayes, param
