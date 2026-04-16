import numpy as np
from src.config import Config
from src.simulation import Simulation
from src.noise import add_noise
from src.layers import create_layers_from_interfaces
from src.utilities import estimate_neff
from sampling.posterior import FWIPosterior


def prepare_synthetic_model():
    config = Config(
        n_receivers=4,
        x_min=0.0,
        x_max=1500.0,
        z_rec=75.0,
        z_src=70.0,
        x_src=50.0,
        nq_prop=64,
        nq_evan=128,
        f0=10.0,
        total_time=1.024,
        delay=0.15,
        epsilon=1.5,
        free_surface=False,
        nfft_pad_factor=4,
        source_deriv=True,
    )

    vp_true = np.array([1500.0, 3000.0, 1800.0, 3500.0])
    z_int = np.array([0.0, 100.0, 250.0, 400.0, 700.0])
    rho = np.array([1200.0, 2100.0, 2000.0, 2200.0])

    layers = create_layers_from_interfaces(z_int, vp_true, rho)
    sim = Simulation(config)

    # 2. Generate Synthetic "Observed" Data
    d_clean, _ = sim.forward(layers)
    d_obs, std_noise = add_noise(d_clean.squeeze(), noise_level=0.1)

    # 3. Prior Parameters
    mu_prior = np.array([4000.0, 4000.0, 4000.0])
    cov_prior = np.diag([500**2, 500**2, 500**2])

    # Calculate tempering factor
    results = estimate_neff(d_clean.squeeze())
    beta = results["n_eff"] / d_obs.size
    print("beta", beta)

    # Initialize the posterior object
    bayes = FWIPosterior(
        d_obs,
        layers,
        sim,
        mu_prior,
        cov_prior,
        std_noise=float(std_noise),
        beta=1.0,
    )

    return bayes
