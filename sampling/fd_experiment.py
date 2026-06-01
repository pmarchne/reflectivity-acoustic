from pathlib import Path
import numpy as np

from src.io_utils import read_fd_observations
from src.config import Config
from src.simulation import Simulation
from src.noise import add_noise_snr
from src.layers import create_layers_from_interfaces
from src.plot.plot_tools import plot_seismogram

from sampling.posterior import FWIPosterior
from sampling.parametrization import ModelParameterization


def compute_receiver_nrmse(obs, fwd):
    """Vectorized NRMSE calculation."""
    rmse = np.sqrt(np.mean((obs - fwd) ** 2, axis=1))
    data_range = np.max(obs, axis=1) - np.min(obs, axis=1)
    return rmse / (data_range + 1e-10)


def prepare_fd_model(file_path="path", seed=42, debug=False):
    """
    Sets up the Bayesian FWI model using external FD observations.
    """
    # Trace indices to extract from the binary FD file
    indices = [28, 32, 36, 40, 44, 48, 52, 56]
    # 1. Configuration matching the FD acquisition
    config = Config(
        n_receivers=57,
        x_min=0.0,
        x_max=700.0,
        z_rec=75.0,
        z_src=50.0,
        x_src=100.0,
        nq_prop=256,
        nq_evan=512,
        f0=10.0,
        ind_traces=indices,
        total_time=1.024,
        delay=0.1,
        epsilon=1.5,
        source_deriv=True,
        free_surface=False,
        nfft_pad_factor=1,
    )

    # Geologic Model
    z_int = np.array([0.0, 100.0, 200.0, 275.0, 375.0, 400.0, 500.0, 550.0, 700.0])
    vps_ref = np.array([1505.0, 2000.0, 2200.0, 4400.0, 1900.0, 3800.0, 2900.0, 3500.0])
    rhos = np.full_like(vps_ref, 2000.0)

    # 2. Physics & Layers
    sim = Simulation(config)
    layers = create_layers_from_interfaces(z_int, vps_ref, rhos)

    # 3. Load & Pre-process FD Observations
    # nt_ref: samples in binary file | nt_cal: samples for our inversion
    d_obs_fd, global_scale = read_fd_observations(
        file_path=Path(file_path),
        nr=57,
        nt_ref=2048,
        nt_cal=sim.param.nt,
        total_time=config.total_time,
        ind_traces=indices,
        normalize=True,
    )

    # std_noise will be relative to the normalized peak (1.0)
    d_obs_final, std_noise = add_noise_snr(d_obs_fd, snr_db=15, seed=seed)

    # 4. Consistency Check
    d_fwd = sim.forward(layers)
    scale = np.max(d_fwd.squeeze())
    d_fwd = (
        d_fwd.squeeze() / scale
    )  # Normalize s.t both observed and calculated data have the same scale at the 'true' vp solution

    nrmse = compute_receiver_nrmse(d_obs_fd, d_fwd)
    print("--- Data Summary ---")
    print(f"FD Scale Factor: {global_scale:.2e}")
    print(f"Reflectivity Scale Factor: {scale:.2e}")
    print(f"Mean NRMSE: {np.mean(nrmse)*100:.2f}%")
    print(f"Estimated Noise Std: {std_noise:.4f}")

    if debug is True:
        plot_seismogram(d_obs_final.T, sim.acq.xr, sim.param.time, vmin=-0.15, vmax=0.15, ncolors=256, figsize=(5,5))
        plot_seismogram(d_fwd.T, sim.acq.xr, sim.param.time, vmin=-0.15, vmax=0.15, ncolors=256, figsize=(5,5))
        vps_tmp = np.array([1505.0, 1403.0, 1749.0, 1819.0, 3979.0, 3000.0, 1265.0, 3200.0])
        lays = create_layers_from_interfaces(z_int, vps_tmp, rhos)
        d_tmp = sim.forward(lays)
        d_tmp = d_tmp.squeeze() / scale
        plot_seismogram(d_tmp.T, sim.acq.xr, sim.param.time, vmin=-0.15, vmax=0.15, ncolors=256, figsize=(5,5))

    # 5. Bayesian Setup
    # Prior for Vp
    mu = 3000.0 * np.ones(len(vps_ref) - 1)
    sigma = 1500.0 * np.ones(len(mu))
    cov = np.diag(sigma**2)

    # Calculate effective samples
    beta = config.total_time * config.f0 / sim.param.nt
    print(f"Tempering factor beta: {beta:.4f}")

    param = ModelParameterization(
        layers,
        invert_vp=True,
        invert_h=False,
        startv=1,
        prior_mode='gaussian',
        mu=mu,
        cov=cov
    )

    # Initialize the posterior object
    bayes = FWIPosterior(
        dobs=d_obs_final,
        parametrization=param,
        sim=sim,
        prior_mu=mu,
        prior_cov=cov,
        std_noise=float(std_noise),
        beta=beta,  # Start with 1.0
        scale_factor=scale,
    )
    return bayes, param


if __name__ == "__main__":
    model = prepare_fd_model()
    print("Success: Bayesian Model ready for UltraNest.")
