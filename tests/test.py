import os
import sys
sys.path.append(os.path.abspath(os.path.join("../")))

from pathlib import Path
import numpy as np
from src.utilities import ricker_wavelet
from src.utilities import timer
from src.fortran.reflectivity_benchmark import reflectivity_q, fortran_reflectivity
from src.parameters import Parameters
from src.acquisition import Acquisition
from src.forward import forward

def test_ricker_spectrum():
    total_time = 1.2  # simulation time
    f0 = 6.0
    f_max = 8.0 * f0
    dt = 1/(2.*f_max)
    nt = int(total_time / dt) + 1
    time = np.arange(nt) * dt
    # zero-pad to the next power of 2
    nfft = 2**int(np.ceil(np.log2(nt)))

    delay = 0.2
    s_t = ricker_wavelet(time, f0, delay)
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


def test_FD_reflectivity():
    base_dir = Path(__file__).resolve().parents[1]  # project/
    file_path = base_dir / "FD_comparison" / "fsismos_P0000"
    seismo_fd = np.fromfile(file_path, dtype=np.float32)

    nt_toy = 2048
    nr = 57
    seismo_fd = seismo_fd.reshape((nr, nt_toy))
    seismo_fd = seismo_fd.T
    total_time = 1.024
    time_toy = np.linspace(0, total_time, nt_toy)
    seismo_fd = seismo_fd / np.max(np.abs(seismo_fd))

    # example from TOYxDAC_TIME
    z_interfaces = np.array([0.0, 100.0, 200.0, 250.0, 350.0, 450.0, 550.0, 650.0, 700.0])
    vp = np.array([1505.0, 1603.0, 1749.0, 2019.0, 2179.0, 1900.0, 2265.0, 3281.0])
    rho = np.array([2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0])
    widths = np.diff(z_interfaces)

    layers = [
        (widths[i], vp[i], rho[i])
        for i in range(len(vp))
    ]
    
    f0 = 10.0           # Ricker central frequency (Hz)
    f_max = 8.0 * f0    # practical Ricker cutoff
    # Nyquist frequency for f_max
    dt = 1/(2.*f_max)
    nfft = 2**int(np.ceil(np.log2(total_time/dt)))
    nfft *= 8  # zero-padding to ensure late wrap-around
    nt = int(total_time / dt) + 1

    epsilon = 1.0
    delay = 0.1
    param = Parameters(total_time=total_time, nt=nt, f0=f0, nfft=nfft, epsilon=epsilon, delay=delay)

    # x and z positions of sources
    xs, zs = 100.0, 50.0
    sources = [(xs, zs)]
    x_receivers = np.linspace(0.0, 700.0, 57) # offsets
    receivers = [(x, 75.0) for x in x_receivers]
    acq = Acquisition(sources, receivers)

    d_cal = forward(layers, acq, param, nq_prop=1024, free_surface=1, timing=True)
    ref = d_cal[0, :, :]

    nt_ref = len(param.time)
    ref = ref.reshape((nr, nt_ref))
    ref_normed = ref / np.max(np.abs(ref))

    mask = time_toy > 0.6 # late arrivals for error analysis
    err, errL2 = np.zeros(nr), np.zeros(nr)
    for ind in range(nr):
        trace_ref = ref_normed[ind, :]
        trace_ref_interp = np.interp(time_toy, param.time, trace_ref)
        trace = seismo_fd[:, ind]
        residual = trace_ref_interp - trace
        err[ind] = np.max(np.abs(residual))
        errL2[ind] = np.sqrt(np.sum(residual[mask]**2))

    max_recv_err = np.max(err)
    print(f"max abs error reflectivity vs TOYxDAC: {max_recv_err:.3e}")
    max_recv_errL2 = np.max(errL2)
    print(f"max L2 error reflectivity vs TOYxDAC: {max_recv_errL2:.3e}")
    
    assert max_recv_err < 5e-2, f"Max error too high: {max_recv_err}"
    assert max_recv_errL2 < 1e-1, f"Max L2 error too high: {max_recv_errL2}"