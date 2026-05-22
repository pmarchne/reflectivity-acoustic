import numpy as np

def add_noise_snr(d_clean, snr_db, seed=None):
    """
    Add Gaussian noise to data.
    Input:
        d_clean : a set of seismic traces on receivers
    Returns:
        d_obs, std_noise
    """
    rng = np.random.default_rng(seed)

    signal_power = np.mean(d_clean**2)
    noise_power = signal_power / (10**(snr_db / 10))

    std_noise = np.sqrt(noise_power)

    noise = rng.normal(0.0, std_noise, size=d_clean.shape)

    return d_clean + noise, std_noise
