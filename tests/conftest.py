import numpy as np
import pytest
from src.config import Config
from src.simulation import Simulation
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
def sim_fd(config_fd):
    return Simulation(config_fd)


@pytest.fixture
def param_fd(sim_fd):
    return sim_fd.param


@pytest.fixture
def fd_geometry(sim_fd):
    return sim_fd.acq


@pytest.fixture
def sim_small(config_small):
    return Simulation(config_small)


@pytest.fixture
def param_small(sim_small):
    return sim_small.param


@pytest.fixture
def sim_fft(config_fft):
    return Simulation(config_fft)


@pytest.fixture
def param_fft(sim_fft):
    return sim_fft.param


@pytest.fixture
def layered_model():
    z_interfaces = np.array([0.0, 100.0, 200.0, 250.0, 350.0,
                             450.0, 550.0, 650.0, 700.0])
    vp = np.array([1505.0, 1603.0, 1749.0, 2019.0, 2179.0,
                   1900.0, 2265.0, 3281.0])
    rho = np.full_like(vp, 2000.0)
    layers = create_layers_from_interfaces(z_interfaces, vp, rho)
    return layers
