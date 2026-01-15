import numpy as np
from utilities import get_kz_chunk
from layers import to_arrays

import time
import matplotlib.pyplot as plt

try:
    from numba import njit, prange
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False

def get_kz_chunk(omega, c, kx_chunk) -> np.ndarray:
    omega = np.asarray(omega)  # shape (Nw,)
    kx = np.asarray(kx_chunk)[None, :]  # shape (1, chunk)
    k0 = omega[:, None] / c  # shape (Nw, 1)

    kz2 = k0**2 - kx**2  # shape (Nw, chunk)
    kz = np.sqrt(kz2 + 0j)  # principal branch

    # enforce principal branch (imag(kz) >= 0)
    kz = np.where(np.imag(kz) < 0, -kz, kz)
    return kz

# ----- NumPy version -----
def _reflectivity_numpy(layers, omegas, kx_chunk, zr, zs, free_surface):
    """
    h, vp, rho: from `layers` (list of tuples) length L (top...bottom)
    omegas: (Nw,)
    kx_chunk: (Nw, chunk)
    returns R_chunk: (Nw, chunk) complex128
    """
    h, vp, rho = to_arrays(layers)
    L = len(h)
    Nw, Nkx = kx_chunk.shape
    R = np.zeros((Nw, Nkx), dtype=np.complex128) 
    Omega = np.asarray(omegas)[:, None]  # (Nw,1)

    _, vp_bot, rho_bot = layers[-1]
    kz_next = get_kz_chunk(omegas, vp_bot, kx_chunk) # (Nw, chunk)
    Z_next = (Omega * rho_bot) / kz_next

    # upward recursion
    for ell in range(L - 2, -1, -1):
        depth = h[ell + 1] # thickness of layer below this interface
        vp_cur = vp[ell]
        rho_cur = rho[ell]

        kz_cur = get_kz_chunk(omegas, vp_cur, kx_chunk)
        Z_cur = (Omega * rho_cur) / kz_cur
        
        r = (Z_next - Z_cur) / (Z_next + Z_cur)
        phase = np.exp(1j * (2.0 * kz_next * depth))

        denom = 1.0 + r * R * phase
        R = (r + R * phase) / denom

        kz_next = kz_cur
        Z_next = Z_cur
    
        if free_surface:
            # distances
            z_minus = abs(zr - zs) # direct distance between r and s
            z_plus  = abs(zr + zs) # distance to image source -s
            # direct and image spectral factors (unit source amplitude)
            direct = np.exp(1j * kz_next * z_minus)
            image  = -np.exp(1j * kz_next * z_plus) # sign for pressure-release
            # spectral Green (source->receiver) including free surface (no stack multiple-bounce)
            G_sr = (direct + image)
            # incorporate stack reflectivity R 
            # geometric-sum using the round-trip factor
            roundtrip = R * (-1.0) * np.exp(1j * 2.0 * kz_next * h[0])
            denom = 1.0 - roundtrip
            if (abs(denom.any()) == 0.0):
                denom = 1e-12 + 0j
            mult_factor = 1.0 / denom
            # final spectral reflectivity
            R_spec = R * mult_factor  
            R = R_spec * G_sr
            R -= direct  # remove direct wave if needed
    return R

# ----- Optional Numba version -----
if _NUMBA_AVAILABLE:
    from numba import njit
    @njit(parallel=True, fastmath=True)
    def _reflectivity_numba_core(h, vp, rho, omegas, kx_chunk, zr, zs, free_surface):
        """
        h_arr, c_arr, rho_arr: 1D arrays of length L (top..bottom)
        omegas: (Nw,)
        kx_chunk: (Nw, chunk)
        returns R_chunk: (Nw, chunk) complex128
        """
        L = h.size
        Nw, chunk = kx_chunk.shape
        R_chunk = np.zeros((Nw, chunk), dtype=np.complex128)

        # For each frequency and each kx column compute upward recursion (parallel over columns)
        for j in range(chunk):
            # For each frequency compute reflectivity column-wise
            for i in range(Nw):
                omega = omegas[i]
                # bottom half-space: start with R = 0
                # compute kz and Z for bottom layer
                c_bot = vp[L - 1]
                rho_bot = rho[L - 1]
                k0 = omega / c_bot
                kx = kx_chunk[i, j]
                kz_next = complex(0.0, 0.0)
                # compute kz_next robustly:
                kz2 = k0 * k0 - kx * kx
                kz_next = np.sqrt(kz2 + 0j)
                if (kz_next.imag < 0.0):
                    kz_next = -kz_next
                Z_next = (omega * rho_bot) / kz_next

                R = 0.0 + 0.0j  # bottom half-space reflectivity
                # upward recursion
                # ell indexes layers from L-2 down to 0 (interface between ell and ell+1)
                for ell in range(L - 2, -1, -1):
                    h_below = h[ell + 1]   # thickness of layer below interface
                    c_cur = vp[ell]
                    rho_cur = rho[ell]
                    # print("vp_cur", c_cur, "rho_cur", rho_cur, "h_curr", h_below)
                    # compute kz_cur and Z_cur
                    k0_cur = omega / c_cur
                    kz2_cur = k0_cur * k0_cur - kx * kx
                    kz_cur = np.sqrt(kz2_cur + 0j)
                    if (kz_cur.imag < 0.0):
                        kz_cur = -kz_cur
                    Z_cur = (omega * rho_cur) / kz_cur
                    # Fresnel at this interface (Z_cur above, Z_next below)
                    r = (Z_next - Z_cur) / (Z_next + Z_cur)
                    # phase through layer below (kz_next)
                    phase = np.exp(1j * 2.0 * kz_next * h_below)
                    denom = 1.0 + r * R * phase
                    R = (r + R * phase) / denom
                    # shift for next iteration
                    kz_next = kz_cur
                    Z_next = Z_cur

                
                if free_surface:
                    # distances
                    z_minus = abs(zr - zs) # direct distance between r and s
                    z_plus  = abs(zr + zs) # distance to image source -s
                    # direct and image spectral factors (unit source amplitude)
                    direct = np.exp(1j * kz_next * z_minus)
                    image  = -np.exp(1j * kz_next * z_plus) # sign for pressure-release
                    # spectral Green (source->receiver) including free surface (no stack multiple-bounce)
                    G_sr = (direct + image)
                    # incorporate stack reflectivity R 
                    # geometric-sum using the round-trip factor
                    roundtrip = R * (-1.0) * np.exp(1j * 2.0 * kz_next * h[0])
                    denom = 1.0 - roundtrip
                    if (abs(denom) == 0.0):
                        denom = 1e-12 + 0j
                    mult_factor = 1.0 / denom
                    # final spectral reflectivity
                    R_spec = R * mult_factor  
                    R = R_spec * G_sr
                    R -= direct  # remove direct wave if needed

                R_chunk[i, j] = R
        return R_chunk

def reflectivity(layers, omegas, thetas, zr, zs, mode="k0", use_numba=True, fs=False):
    
    if use_numba and not _NUMBA_AVAILABLE:
        print("Warning: Numba not installed, falling back to NumPy version.")
        use_numba = False

    # unpack layers
    h, vp, rho = to_arrays(layers)
    vp_top = vp[0]  # top-layer velocity
    # shapes
    omegas = np.asarray(omegas)
    thetas = np.asarray(thetas)
    # compute kx(w,θ)
    if mode == "k0":
        # k0 = omega/vp_top
        k0 = omegas[:, None] / vp_top
        kx_chunk = k0 * np.sin(thetas)[None, :]
    elif mode == "psi":
        k0 = omegas[:, None] / vp_top
        kx_chunk = k0 * np.cosh(thetas)[None, :]
    else:
        raise ValueError("mode must be 'k0' or 'psi'")
    
    # call reflectivity engine
    start = time.time()
    if use_numba:
        # Call numba version (to be implemented)
        R = _reflectivity_numba_core(h, vp, rho, omegas, kx_chunk, zr, zs, free_surface=fs)
    else:
        R = _reflectivity_numpy(layers, omegas, kx_chunk, zr, zs, free_surface=fs).squeeze()
    end = time.time()
    print(f"elapsed: {end-start:.2f} s")
    #R = _reflectivity_numpy(layers, omegas, kx_chunk)
    return R

# Example usage
if __name__ == "__main__":
    layers = [
        (100.0, 1500.0, 1800.0),
        (250.0, 1900.0, 2000.0),
        (350.0, 1700.0, 2200.0),
        (500.0, 2000.0, 2400.0),
    ]

    freqs = np.linspace(0.1, 100.0, 1024)
    omegas = 2.0 * np.pi * freqs

    kx_max = 0.5
    Nkx = 2048*8
    kx_vals = np.linspace(-kx_max, kx_max, Nkx)
    kx_grid = np.tile(kx_vals[None, :], (omegas.size, 1))

    #use_numba = True
    #chunk = 128

    h, vp, rho = to_arrays(layers)
    R_numpy = _reflectivity_numpy(layers, omegas, kx_grid)
    start = time.time()
    R_numba = _reflectivity_numba_core(h, vp, rho, omegas, kx_grid, 70., 80., free_surface=False)
    end = time.time()
    print(f"elapsed: {end-start:.2f} s")

    start = time.time()
    R_numba = _reflectivity_numba_core(h, vp, rho, omegas, kx_grid, 70., 80., free_surface=False)
    end = time.time()
    print(f"elapsed: {end-start:.2f} s")
