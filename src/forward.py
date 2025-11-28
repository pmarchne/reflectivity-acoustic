import numpy as np
import time
from src.parameters import Parameters
from src.acquisition import Acquisition
from src.utilities import ricker_wavelet, green2d_flat
from src.quadrature.gauss_cheby import integral_kx_quadrature_numba
from src.quadrature.filon_Sommerfeld import Sommerfeld_integral

def forward(layers, acq: Acquisition, param: Parameters, free_surface=False, nq_prop=1024, nq_evan=512):
    """
    Forward modeling to compute predicted data d_cal
    using the reflectivity method with optional free surface.
    
    Args:
        layers: list of tuples (thickness, vp, rho)
        acq: Acquisition object with xs, zs, xr, zr
        param: Parameters object with dt, nfft, nt, time, freq
        free_surface: whether to include free surface at z=0
    
    Returns:
        d_cal: array (Ns, Nr, nt) of predicted time-domain traces
    """
    freqs = np.fft.rfftfreq(param.nfft, param.dt)
    epsilon = 1.0 # complex frequency damping
    print("max epsilon =", np.log(50)/param.total_time)
    omegas = 2.0 * np.pi * freqs + 1j * epsilon
    vp_top = layers[0][1]

    s_t, delay = ricker_wavelet(param.time, param.freq)
    s_t = s_t * np.exp(-epsilon * param.time)
    #print(f"Ricker wavelet initialized with delay: {delay} s")
    # ---- FFT in omega using the +i w t convention ----
    s_w = np.conj(np.fft.rfft(np.conj(s_t), n=param.nfft)) 

    # ---- Acquisition geometry ----
    xs, zs, xr, zr = acq.xs, acq.zs, acq.xr, acq.zr
    Ns, Nr = xs.size, xr.size

    # ---- kx quadrature ----
    kx_factor = 8.
    start = time.time() # z_travel
    R_map = Sommerfeld_integral(
        layers, omegas, xs, zs, xr, zr,
        nq_prop, nq_evan, kx_max_factor=kx_factor)
    #R_map = integral_kx_quadrature_numba(
    #    layers, omegas, xs, zs, xr, zr,
    #    nq_prop, nq_evan, kx_max_factor=kx_factor, chunk=256, fs=free_surface)
    end = time.time()
    print(f"kx quadrature elapsed: {end-start:.2f} s")
    # Multiply by hanning window
    #R_map = np.half_haning() # 0.5 *( 1+np.cos(omega*pi/Om) )
    # ---- Flatten for source-receiver pairs ----
    Ns, Nr, Nw = R_map.shape
    R_flat = R_map.reshape((Ns*Nr, Nw))

    travel_xy = np.sqrt((xs[:, None] - xr[None, :])**2 + (zs[:, None] - zr[None, :])**2)
    distances_flat = travel_xy.ravel()
    G_flat = green2d_flat(omegas, vp_top, distances_flat)

    taper_freq = np.hanning(2*Nw)[Nw:] 

    T_flat = (R_flat+G_flat) * s_w[None, :] * np.exp(-1j * delay * omegas)[None, :]
    T_flat *= taper_freq[None, :]
    #T_flat = (R_flat) * s_w[None, :] #* np.exp(-1j * delay * omegas)[None, :]
    # T_flat *= 1j*np.real(omegas) # ricker source time-derivative !
    # ---- Inverse FFT to time domain ----
    traces_full = np.conj(np.fft.irfft(np.conj(T_flat), n=param.nfft, axis=1))
    traces_cut = traces_full[:, :param.nt] * np.exp(epsilon * param.time)

    # ---- Reshape to (Ns, Nr, nt) ----
    d_cal = traces_cut.reshape((Ns, Nr, param.nt))
    return d_cal
