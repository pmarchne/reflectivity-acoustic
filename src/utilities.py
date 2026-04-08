import numpy as np
from contextlib import contextmanager
import time
from src.parameters import Parameters

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
    ricker = (1 - 2 * (np.pi * f0 * tau)**2) * np.exp(-(np.pi * f0 * tau)**2)
    return ricker

def source_frequency(param: Parameters):
    """
    convert source from time to frequency domain
    param: Parameters object
    """
    omegas = param.create_frequencies()
    source_time = ricker_wavelet(param.time, param.f0, param.delay)
    #print(f"Ricker wavelet initialized with delay: {delay} s")
    source_time *= np.exp(-param.epsilon * param.time) # apply damping in time domain 
    # ---- FFT in omega using the +i w t convention ----
    source_freq = np.conj(np.fft.rfft(source_time, n=param.nfft))

    return source_freq, omegas

def inverse_fft_signal(signal_freq, param, windowing=None):
    """
    Inverse FFT to time domain using +i w t convention.
    Parameters:
        signal_freq : frequency domain signal (N_traces, N_w).
    Returns:
        signal_time : time domain signal (N_traces, N_time).
    """
    #taper_freq = np.hanning(2*Nw)[Nw:]
    #T_flat *= taper_freq[None, :] 
    traces = np.conj(np.fft.irfft(np.conj(signal_freq), n=param.nfft, axis=1))
    traces_cut = traces[:, :param.nt] * np.exp(param.epsilon * param.time)
    return traces_cut

def low_freq_taper(omegas, omega_min):
    """
    Smoothly suppress frequencies below omega_min.
    """
    taper = np.ones_like(omegas)

    mask = omegas < omega_min
    taper[mask] = 0.5 * (1 - np.cos(np.pi * omegas[mask] / omega_min))

    return taper

def get_kz(omega, vp, p) -> np.ndarray:
    omega = np.asarray(omega)[:, None]  # shape (Nw,)
    p = np.asarray(p)[None, :]  # shape (Np,)
    kz2 = omega**2 * (1.0 / vp**2 - p**2)
    kz = np.sqrt(kz2 + 0j)  # principal branch
    # enforce principal branch (imag(kz) >= 0)
    kz = np.where(np.imag(kz) < 0, -kz, kz)
    return kz

def get_critical_angles(layers, verbose=True):
    """
    Get critical angles from layer 1 to all deeper layers and return the minimum.
    """
    v1 = layers[0][1]
    critical_angles = []
    
    for i in range(1, len(layers)):
        v_target = layers[i][1]
        if v1 < v_target:
            theta_c_deg = np.degrees(np.arcsin(v1 / v_target))
            critical_angles.append((i+1, v_target, theta_c_deg))
            if verbose:
                print(f"Layer 1 -> Layer {i+1}: {theta_c_deg:.2f}° (v={v_target:.1f} m/s)")
    
    min_angle = min(ca[2] for ca in critical_angles) if critical_angles else None
    
    if verbose and min_angle:
        print(f"Minimum critical angle: {min_angle:.2f}°")
    
    return min_angle


