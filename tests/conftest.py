import numpy as np
import pytest
from src.config import Config
from src.builders import build_problem
from src.layers import create_layers_from_interfaces


@pytest.fixture
def config_small():
    return Config(total_time=1.2, f0=6.0, epsilon=0.0, delay=0.2)


@pytest.fixture
def config_fft():
    return Config(total_time=1.2, f0=9.0, epsilon=0.95, delay=0.2)


@pytest.fixture
def config_fd():
    return Config(
        n_receivers=57,
        noise_level=0.1,
        x_min=0.0,
        x_max=700.0,
        z_rec=75.0,
        z_src=50.0,
        x_src=100.0,
        nq_prop=1024,
        f0=10.0,
        total_time=1.024,
        delay=0.1,
        nfft_pad_factor=4,
        free_surface=True,
        epsilon=1.0,
    )


@pytest.fixture
def param_fd(config_fd):
    param, _ = build_problem(config_fd)
    return param


@pytest.fixture
def fd_geometry(config_fd):
    _, acq = build_problem(config_fd)
    return acq


@pytest.fixture
def param_small(config_small):
    param, _ = build_problem(config_small)
    return param


@pytest.fixture
def param_fft(config_fft):
    param, _ = build_problem(config_fft)
    return param


@pytest.fixture
def layered_model():
    z_interfaces = np.array([0.0, 100.0, 200.0, 250.0, 350.0,
                             450.0, 550.0, 650.0, 700.0])
    vp = np.array([1505.0, 1603.0, 1749.0, 2019.0, 2179.0,
                   1900.0, 2265.0, 3281.0])
    rho = np.full_like(vp, 2000.0)
    layers = create_layers_from_interfaces(z_interfaces, vp, rho)
    return layers
