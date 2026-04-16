import numpy as np
from contextlib import contextmanager
import time

from src.parameters import Parameters
from src.config import Config
from src.layers import to_arrays


@contextmanager
def timer(label: str, enabled: bool = True):
    if enabled:
        start = time.time()
    yield
    if enabled:
        elapsed = time.time() - start
        print(f"{label} elapsed: {elapsed:.2f} s")


def ricker_wavelet(t, f0, t0=0.2):
    """
    Parameters:
        t : time variable [s].
        f0 : central frequency [Hz].
        t0 : wavelet time shift [s].
    Returns:
        ricker : wavelet values.
    """
    tau = t - t0
    ricker = (1 - 2 * (np.pi * f0 * tau) ** 2) * np.exp(-((np.pi * f0 * tau) ** 2))
    return ricker


def source_frequency(param: Parameters, config: Config):
    """Convert source wavelet from time to frequency domain."""
    source_time = ricker_wavelet(param.time, config.f0, config.delay)
    # Apply damping in time domain
    source_time *= np.exp(-config.epsilon * param.time)
    # FFT using the +i w t convention
    source_freq = np.conj(np.fft.rfft(source_time, n=param.nfft)) * param.dt
    return source_freq


def inverse_fft_signal(signal_freq, param: Parameters, config: Config):
    """
    Inverse FFT to time domain using +i w t convention.

    Parameters:
        signal_freq : frequency domain signal (N_traces, N_w).
    Returns:
        signal_time : time domain signal (N_traces, N_time).
    """
    traces = np.conj(np.fft.irfft(np.conj(signal_freq), n=param.nfft, axis=1))
    traces_cut = traces[:, : param.nt] * np.exp(config.epsilon * param.time)
    return traces_cut / param.dt


def adjoint_inverse_fft_signal(signal_time, param, config):
    """Adjoint of inverse_fft_signal: maps time-domain residuals to frequency domain."""
    temp = signal_time * np.exp(config.epsilon * param.time)

    padded = np.zeros((temp.shape[0], param.nfft), dtype=np.float64)
    padded[:, : param.nt] = temp

    s_w = np.conj(np.fft.rfft(np.conj(padded), n=param.nfft, axis=1))

    if param.nfft % 2 == 0:
        s_w[:, 1:-1] *= 2.0
    else:
        s_w[:, 1:] *= 2.0

    s_w /= param.nfft * param.dt
    return s_w


def low_freq_taper(omegas, omega_min):
    """Smoothly suppress frequencies below omega_min."""
    taper = np.ones_like(omegas)
    mask = omegas < omega_min
    taper[mask] = 0.5 * (1 - np.cos(np.pi * omegas[mask] / omega_min))
    return taper


def get_critical_angles(layers):
    """
    Compute the critical angle at each interface.

    The critical angle at interface i→i+1 is defined as:
        θ_c = arcsin(vp[i] / vp[i+1])
    and only exists when vp[i+1] > vp[i] (velocity increase downward).

    Parameters:
        layers: list of Layer objects or (h, vp, rho) tuples.
    Returns:
        list of (interface_index, critical_angle_degrees or None).
    """

    _, vps, _ = to_arrays(layers)
    results = []
    for i in range(len(vps) - 1):
        v1, v2 = vps[i], vps[i + 1]
        if v2 > v1:
            theta_c = np.degrees(np.arcsin(v1 / v2))
            print(
                f"Interface {i}→{i+1}: vp {v1:.0f}→{v2:.0f} m/s, "
                f"critical angle = {theta_c:.2f}°"
            )
            results.append((i, theta_c))
        else:
            print(
                f"Interface {i}→{i+1}: vp {v1:.0f}→{v2:.0f} m/s, "
                f"no critical angle (velocity decrease)"
            )
            results.append((i, None))
    return results


def get_kz(omega, vp, p) -> np.ndarray:
    omega = np.asarray(omega)[:, None]  # shape (Nw,)
    p = np.asarray(p)[None, :]  # shape (Np,)
    kz2 = omega**2 * (1.0 / vp**2 - p**2)
    kz = np.sqrt(kz2 + 0j)  # principal branch
    # Enforce Im(kz) >= 0
    kz = np.where(np.imag(kz) < 0, -kz, kz)
    return kz


def estimate_neff(residual):
    """
    Estimates the effective number of samples (N_eff) from a 2D residual array.
    Assumes residual shape is (n_receivers, n_time).
    """
    n_receivers, n_time = residual.shape

    def get_autocorr(x):
        """Computes the normalized autocorrelation."""
        if np.all(x == 0):
            return np.zeros_like(x)
        x_centered = x - np.mean(x)
        r = np.correlate(x_centered, x_centered, mode="full")
        r = r[r.size // 2 :]
        return r / r[0]

    def get_integrated_tau(rho):
        """Estimates integrated correlation time, stopping at the first negative value."""
        # Find the first index where rho <= 0
        neg_indices = np.where(rho[1:] <= 0)[0]
        stop_idx = neg_indices[0] + 1 if neg_indices.size > 0 else len(rho)

        # Sum only the positive leading part of the autocorrelation
        return 1 + 2 * np.sum(rho[1:stop_idx])

    # 1. Calculate Temporal N_eff (Across Time for each receiver)
    taus_time = []
    for i in range(n_receivers):
        rho = get_autocorr(residual[i, :])
        taus_time.append(get_integrated_tau(rho))

    tau_mean_time = np.mean(taus_time)
    n_eff_time = n_time / tau_mean_time

    # 2. Calculate Spatial N_eff (Across receivers for each time sample)
    taus_space = []
    for t in range(n_time):
        rho = get_autocorr(residual[:, t])
        taus_space.append(get_integrated_tau(rho))

    tau_mean_space = np.mean(taus_space)
    n_eff_receivers = n_receivers / tau_mean_space

    # 3. Total Effective Samples
    total_n_eff = n_eff_time * n_eff_receivers

    return {
        "n_eff": total_n_eff,
        "n_eff_time": n_eff_time,
        "n_eff_receivers": n_eff_receivers,
        "tau_time": tau_mean_time,
        "tau_space": tau_mean_space,
    }
