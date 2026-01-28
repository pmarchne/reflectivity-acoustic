from dataclasses import dataclass
import numpy as np
from src.parameters import Parameters
from src.acquisition import Acquisition


@dataclass
class FWIConfig:
    # Forward modeling
    fs: bool = True
    total_time: float = 2.0
    f0: float = 5.0
    epsilon: float = 0.5
    nq_prop: int = 256

    # Acquisition
    Nr: int = 24
    x_min: float = 100.0
    x_max: float = 4000.0
    z_src: float = 76.0
    z_rec: float = 76.0
    x_src: float = 30.0

    # Noise
    noise_level: float = 0.1


def build_problem(config: FWIConfig):
    # time sampling
    f_max = 8.0 * config.f0
    dt = 1 / (2 * f_max)
    nt = int(config.total_time / dt) + 1
    nfft = 2**int(np.ceil(np.log2(config.total_time/dt))) * 2

    param = Parameters(
        total_time=config.total_time,
        nt=nt,
        f0=config.f0,
        nfft=nfft,
        epsilon=config.epsilon
    )

    # acquisition
    sources = [(config.x_src, config.z_src)]
    xs = np.linspace(config.x_min, config.x_max, config.Nr)
    receivers = [(x, config.z_rec) for x in xs]
    acq = Acquisition(sources, receivers)

    return param, acq
