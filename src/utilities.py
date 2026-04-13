import numpy as np
from contextlib import contextmanager
import time

from src.parameters import Parameters
from src.config import Config


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


def get_kz(omega, vp, p) -> np.ndarray:
    omega = np.asarray(omega)[:, None]  # shape (Nw,)
    p = np.asarray(p)[None, :]  # shape (Np,)
    kz2 = omega**2 * (1.0 / vp**2 - p**2)
    kz = np.sqrt(kz2 + 0j)  # principal branch
    # Enforce Im(kz) >= 0
    kz = np.where(np.imag(kz) < 0, -kz, kz)
    return kz
