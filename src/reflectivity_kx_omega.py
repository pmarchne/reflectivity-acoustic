import numpy as np
from utilities import get_kz_chunk
from layers import layers_to_arrays

import time
import matplotlib.pyplot as plt

try:
    from numba import njit
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False

# ----- NumPy version -----
def _reflectivity_numpy(layers, omegas, kx_chunk, free_surface=False):
    """
    h, vp, rho: from `layers` (list of tuples) length L (top...bottom)
    omegas: (Nw,)
    kx_chunk: (Nw, chunk)
    returns R_chunk: (Nw, chunk) complex128
    """
    h, vp, rho = layers_to_arrays(layers)
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
        # reflection coefficient between air (Z_air) and top layer (Z_next)
        Z_air = 1.225 * 343.0
        r_surface = (Z_air - Z_next) / (Z_air + Z_next)   # equals -1 if Z_air == 0
        # combine surface reflection with downward reflectivity R (no extra phase above surface)
        R = (r_surface + R) / (1.0 + r_surface * R)

    return R

# ----- Optional Numba version -----
if _NUMBA_AVAILABLE:
    from numba import njit
    @njit(parallel=True, fastmath=True)
    def _reflectivity_numba_core(h, vp, rho, omegas, kx_chunk, free_surface=False):
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

                if free_surface: # refelctivity map is now synthetized at z=0 !
                    # Z_air = 1.225 * 343.0
                    r_surf = -1.0 # or (Z_air - Z_next) / (Z_air + Z_next)
                    phase = np.exp(1j * 2.0 * kz_next * h[0])
                    denom = 1.0 - r_surf * R * phase
                    R = r_surf + (R * phase) / denom

                # store
                R_chunk[i, j] = R
        return R_chunk

# ----- interface -----
def get_reflectivity_chunked(layers, omegas, kx_chunk, use_numba=False, free_surface=False):
    """
    Compute surface reflectivity matrix R(omega, kx_chunk).
    Parameters
    ----------
    layers : list of (h, c, rho) tuples
    omegas : array-like, shape (Nw,)
    kx_chunck : array-like, shape (Nw, Nkx_chunck)
    use_numba : bool
        If True, attempt to use Numba version (requires Numba installed).

    Returns
    -------
    R_out : np.ndarray, shape (Nw, Nkx_chunck)
    """
    if use_numba and not _NUMBA_AVAILABLE:
        print("Warning: Numba not installed, falling back to NumPy version.")
        use_numba = False

    if use_numba:
        # Call numba version (to be implemented)
        h, vp, rho = layers_to_arrays(layers)
        return _reflectivity_numba_core(h, vp, rho, omegas, kx_chunk, free_surface=free_surface)
    else:
        return _reflectivity_numpy(layers, omegas, kx_chunk, free_surface=free_surface)
    
# Example usage
if __name__ == "__main__":
    layers = [
        (100.0, 1500.0, 1800.0),
        (250.0, 1900.0, 2000.0),
        (350.0, 1700.0, 2200.0),
        (500.0, 2000.0, 2400.0),
    ]

    freqs = np.linspace(0.1, 100.0, 200)
    omegas = 2.0 * np.pi * freqs

    kx_max = 0.5
    Nkx = 2048
    kx_vals = np.linspace(-kx_max, kx_max, Nkx)
    kx_grid = np.tile(kx_vals[None, :], (omegas.size, 1))

    use_numba = True
    chunk = 128
    # For propagating
    R = np.zeros((len(freqs),Nkx), dtype=np.complex128)
    start = time.time()
    for i0 in range(0, Nkx, chunk):
        i1 = min(i0 + chunk, Nkx)
        R[:, i0:i1] = get_reflectivity_chunked(layers, omegas, kx_grid[:, i0:i1], use_numba=use_numba, free_surface=False)
    end = time.time()
    print(f"Reflectivity computed in {end - start:.2f} seconds.")

    plt.figure(figsize=(6,5))
    plt.imshow(np.abs(R), origin='lower',
               extent=[kx_vals[0], kx_vals[-1], freqs[0], freqs[-1]],
               aspect='auto')
    plt.xlabel('kx (rad/m)')
    plt.ylabel('frequency (Hz)')
    plt.title('Reflectivity magnitude |R(omega,kx)|')
    plt.colorbar(label='|R|')
    plt.tight_layout()
    plt.show()
