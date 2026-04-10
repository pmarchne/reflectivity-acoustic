import numpy as np


def add_noise(d_clean, noise_level, seed=None):
    """
    Add Gaussian noise to data.
    Returns:
        d_obs, std_noise
    """
    rng = np.random.default_rng(seed)
    std_noise = noise_level * np.std(d_clean)  # np.sqrt(np.mean(d_clean**2))
    noise = rng.normal(loc=0.0, scale=std_noise, size=d_clean.shape)
    return d_clean + noise, std_noise