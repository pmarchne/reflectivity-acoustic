import numpy as np
from src.config import Config
from src.simulation import Simulation
from src.noise import add_noise_snr
from sampling.parametrization import ModelParameterization, diagnostic_checks
from src.layers import create_layers, create_layers_from_interfaces
from sampling.posterior import FWIPosterior
from src.plot.plot_tools import plot_seismogram
from src.builders import build_parameters
from src.utilities import timer

def prepare_model_marine(seed, debug=False):
    config = Config(
        n_receivers=8,
        x_min=2000.0,
        x_max=7000.0,
        z_rec=65.0,
        z_src=10.0,
        x_src=500.0,
        nq_prop=712, # 800
        nq_evan=512,
        f0=2., # 2.5
        total_time=7.0,
        delay=0.6, # 0.4,
        epsilon=0.75,
        free_surface=True,
        nfft_pad_factor=2,
        source_deriv=True
    )
    param = build_parameters(config=config)
    print("number of time points", param.nt)
    print("number of fft points", param.nfft)

    z_int = np.array([0.0, 70.0, 1700.0, 2000.0, 2700.0, 4000.0]) 
    vp_ref = np.array([1500.0, 2000.0, 1700.0, 2300.0, 3000.0]) # m/s
    rho_ref = np.full_like(vp_ref, 2000.0) # kg/m3
    h_ref = np.diff(z_int)

    layers = create_layers(h_ref, vp_ref, rho_ref)
    sim = Simulation(config)

    # 2. Generate Synthetic "Observed" Data
    d_clean = sim.forward(layers)
    d_obs, std_noise = add_noise_snr(d_clean.squeeze(), snr_db=5, seed=seed)
    print(f"Estimated Noise Std: {std_noise:.4f}")

    if debug is True:
        plot_seismogram(d_clean.T, sim.acq.xr, sim.param.time, vmin=-0.02, vmax=0.02, ncolors=256, figsize=(5,5))
        plot_seismogram(d_obs.T, sim.acq.xr, sim.param.time, vmin=-0.02, vmax=0.02, ncolors=256, figsize=(5,5))
        z = np.array([0.0, 200.0, 1800.0, 2000.0, 2900.0, 4000.0]) 
        vp = np.array([1500.0, 2500.0, 3700.0, 1800.0, 3000.0])
        lays = create_layers_from_interfaces(z, vp, rho_ref)
        with timer("test"):
            d_tmp = sim.forward(lays)
        plot_seismogram(d_tmp.T, sim.acq.xr, sim.param.time, vmin=-0.1, vmax=0.1, ncolors=256, figsize=(5,5))

        vp = np.array([1500.0, 1200.0, 2700.0, 4800.0, 3000.0])
        lays = create_layers_from_interfaces(z_int, vp, rho_ref)
        with timer("test 2"):
            d_tmp = sim.forward(lays)
        plot_seismogram(d_tmp.T, sim.acq.xr, sim.param.time, vmin=-0.1, vmax=0.1, ncolors=256, figsize=(5,5))

    # Prior Parameters
    #mu_prior = np.array([3000.0, 3000.0, 3000.0, 3000.0, 1000.0, 1000.0, 1000.0])
    #cov_prior = np.diag([800**2, 800**2, 800**2, 800**2, 400**2, 400**2, 400**2])
    mu_prior = np.array([2500.0, 2500.0, 2500.0, 2500.0, 1500.0, 500.0, 500.0])
    cov_prior = np.diag([800**2, 800**2, 800**2, 800**2, 300**2, 300**2, 300**2])

    # Calculate effective samples
    factor = 0.15 # model mismatch inflation factor (e.g. 0.2, 0.5)
    beta = factor*config.total_time * config.f0 / sim.param.nt
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
    param.h_bounds = (50., 2500.)

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

    m_ref = np.array([2000.0, 1700.0, 2300.0, 3000.0, 1630., 300.0, 700.])
    diagnostic_checks(bayes, param.prior_transform, m_ref)

    return bayes, param

if __name__ == "__main__":
    prepare_model_marine(seed=42, debug=True)
