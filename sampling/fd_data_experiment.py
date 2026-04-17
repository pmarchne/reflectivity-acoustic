from pathlib import Path
import numpy as np

from posterior import FWIPosterior
from src.io_utils import read_fd_observations
from src.config import Config
from src.simulation import Simulation
from src.noise import add_noise
from src.utilities import estimate_neff
from src.layers import create_layers_from_interfaces


def compute_receiver_nrmse(obs, fwd):
    """Vectorized NRMSE calculation."""
    rmse = np.sqrt(np.mean((obs - fwd) ** 2, axis=1))
    data_range = np.max(obs, axis=1) - np.min(obs, axis=1)
    return rmse / (data_range + 1e-10)


def prepare_fd_model(file_path="FD_comparison/fsismos_P0000_nofs"):
    """
    Sets up the Bayesian FWI model using external FD observations.
    """
    # Trace indices to extract from the binary FD file
    indices = [8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56]
    # 1. Configuration matching the FD acquisition
    config = Config(
        n_receivers=57,
        x_min=0.0,
        x_max=700.0,
        z_rec=75.0,
        z_src=50.0,
        x_src=100.0,
        nq_prop=64,
        nq_evan=64,
        f0=10.0,
        ind_traces=indices,
        total_time=1.024,
        delay=0.1,
        epsilon=1.5,
        source_deriv=True,
        free_surface=False,
        nfft_pad_factor=4,
    )

    # Geologic Model
    z_int = np.array([0.0, 100.0, 200.0, 250.0, 350.0, 450.0, 550.0, 650.0, 700.0])
    vps_ref = np.array([1505.0, 1603.0, 1749.0, 2019.0, 2179.0, 1900.0, 2265.0, 3281.0])
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
    d_obs_final, std_noise = add_noise(d_obs_fd, noise_level=0.1)

    # 4. Consistency Check
    d_fwd, _ = sim.forward(layers)
    scale = np.max(d_fwd.squeeze())
    d_fwd = (
        d_fwd.squeeze() / scale
    )  # Normalize s.t both observed and calculated data
    # have the same scale at the 'true' vp solution

    nrmse = compute_receiver_nrmse(d_obs_fd, d_fwd)
    print("--- Data Summary ---")
    print(f"FD Scale Factor: {global_scale:.2e}")
    print(f"Reflectivity Scale Factor: {scale:.2e}")
    print(f"Mean NRMSE: {np.mean(nrmse)*100:.2f}%")
    print(f"Estimated Noise Std: {std_noise:.4f}")

    # 5. Bayesian Setup
    # Prior for Vp
    mu = 3000.0 * np.ones(len(vps_ref) - 1)
    sigma = 800.0 * np.ones(len(mu))
    cov = np.diag(sigma**2)

    # Calculate effective samples (for tempering)
    results_neff = estimate_neff(d_obs_final.squeeze())
    beta = results_neff["n_eff"] / d_obs_final.size
    print(f"Suggested Beta (n_eff): {beta:.2f}")

    # Initialize the posterior object
    bayes = FWIPosterior(
        dobs=d_obs_final,
        layers=layers,
        sim=sim,
        prior_mu=mu,
        prior_cov=cov,
        std_noise=float(std_noise),
        beta=beta,  # Start with 1.0
        scale_factor=scale,
    )

    return bayes


if __name__ == "__main__":
    model = prepare_fd_model()
    print("Success: Bayesian Model ready for UltraNest.")
