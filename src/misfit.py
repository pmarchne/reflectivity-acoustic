import numpy as np
from src.parameters import Parameters
from src.acquisition import Acquisition
from src.forward import forward


def l2_misfit(dcal, dobs, std_noise):
    """ L2 misfit between predicted and observed data """
    residual = dcal - dobs  # residual for all source/receiver pairs
    return 0.5 * np.sum(residual**2) / (std_noise**2)


def add_noise(dobs, noise_level=0.02):
    ampli = np.sqrt(np.mean(dobs**2)) # np.max(np.abs(dobs))
    std_noise = noise_level * ampli  # 2% noise
    noise = np.random.normal(
        loc=0.0,
        scale=std_noise,
        size=dobs.shape
    )
    return dobs + noise, std_noise