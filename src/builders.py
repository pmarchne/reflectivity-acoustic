import numpy as np
from src.config import Config
from src.parameters import Parameters
from src.acquisition import Acquisition


def build_parameters(config: Config) -> Parameters:
    f_max = 8.0 * config.f0
    dt = 1.0 / (2.0 * f_max)

    nt = int(config.total_time / dt) + 1

    n = int(np.ceil(np.log2(config.total_time / dt)))
    nfft = 2 ** (n + 1)

    time = np.arange(nt) * dt
    freqs = np.fft.rfftfreq(nfft, d=dt)
    eps_f = 1e-6
    omegas = 2 * np.pi * (freqs + eps_f) + 1j * config.epsilon

    return Parameters(
        dt=dt,
        nt=nt,
        nfft=nfft,
        time=time,
        omegas=omegas,
    )


def build_acquisition(config: Config) -> Acquisition:

    xr = np.linspace(config.x_min, config.x_max, config.n_receivers)
    if config.ind_traces:
        xr = xr[config.ind_traces]

    sources = [(config.x_src, config.z_src)]
    receivers = [(x, config.z_rec) for x in xr]
    return Acquisition(sources, receivers)


def build_problem(config: Config):
    return build_parameters(config), build_acquisition(config)
