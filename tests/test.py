import os
import sys
sys.path.append(os.path.abspath(os.path.join("../")))

import pytest
import numpy as np
from src.utilities import ricker_wavelet
from src.utilities import timer
from src.fortran.reflectivity_benchmark import reflectivity_q, fortran_reflectivity

def test_ricker_spectrum():
    total_time = 1.2  # simulation time
    f0 = 6.0
    f_max = 8.0 * f0
    dt = 1/(2.*f_max)
    nt = int(total_time / dt) + 1
    time = np.arange(nt) * dt
    # zero-pad to the next power of 2
    nfft = 2**int(np.ceil(np.log2(nt)))

    s_t, delay = ricker_wavelet(time, f0)
    freqs = np.fft.rfftfreq(nfft, dt)
    s_w = np.conj(np.fft.rfft(s_t, n=nfft))
    s_w *= dt

    # exact spectrum
    exact = (2./np.sqrt(np.pi)) * freqs**2 / (f0**3) \
        * np.exp(-(freqs/f0)**2) * np.exp(1j * 2.* np.pi * freqs * delay)

    error = np.abs(exact - s_w)
    print("max error", np.max(error))
    assert np.max(error) < 1e-6, f"Max error too high: {error}"

def test_reflectivity_benchmark():
    layers = [
        (100.0, 1500.0, 1800.0),
        (250.0, 1900.0, 2000.0),
        (350.0, 3900.0, 2000.0),
        (400.0, 2900.0, 2000.0),
    ]
    freqs = np.linspace(0.1, 50.0, 1024, dtype=np.complex128)
    omegas = 2.0 * np.pi * freqs + 0.5 * 1j
    Ntheta = 2000
    thetas = np.linspace(0., np.pi, Ntheta, dtype=np.float64)
    p = np.sin(thetas) / layers[0][1]

    print("\n ----- Benchmark ----- ")
    repeats = 3
    for r in range(repeats):
        with timer(f"numpy run {r+1} : ", True):
            R_np = reflectivity_q(layers, omegas, p, free_surface=1, zr=70., zs=80.)
        with timer(f"fortran run {r+1} : ", True):
            R_f = fortran_reflectivity(layers, omegas, p, free_surface=1, zr=70., zs=80.)

    error = np.max(np.abs(R_np - R_f))
    print(f"Max abs error between numpy and fortran: {error:.3e}")
    assert np.max(error) < 1e-8, f"Max error too high: {error}"