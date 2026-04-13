import numpy as np
import time

from src.plot.plot_tools import plot_reflectivity

# Attempt to import the compiled Fortran module
try:
    import src.fortran.reflectivity as reflectivity
    FORTRAN_AVAILABLE = True
    rfmod = reflectivity.reflectivity_mod
except Exception as e:
    print("Fortran module not available:", e)
    FORTRAN_AVAILABLE = False
    rfmod = None

# --- NumPy vectorized implementation ---
def numpy_reflectivity_p(
    layers,
    omegas,
    p,
    free_surface=1,
    zr=70.0,
    zs=80.0
):
    # let complex model parameters for adjoint gradient check
    h, vp, rho = map(
        lambda x: np.asarray(x, dtype=np.complex128),
        zip(*layers)
    )
    omegas = np.asarray(omegas, dtype=np.complex128)
    p = np.asarray(p, dtype=np.float64)
    p2 = p**2

    nlay = len(h)
    nw = omegas.size
    nk = p.size

    R = np.zeros((nw, nk), dtype=np.complex128)
    inv_vp2 = 1.0 / (vp * vp)

    for iw, omega in enumerate(omegas):
        omega2 = omega * omega

        kz = np.zeros((nlay, nk), dtype=np.complex128)
        Z = np.zeros((nlay, nk), dtype=np.complex128)
        Rstep = np.zeros((nlay, nk), dtype=np.complex128)
        Rval = np.zeros(nk, dtype=np.complex128)

        kz[-1] = np.sqrt(omega2 * (inv_vp2[-1] - p2) + 0j)
        kz[-1] = np.where(np.imag(kz[-1]) < 0, -kz[-1], kz[-1])
        Z[-1] = omega * rho[-1] / kz[-1]
        Rstep[-1] = 0.0 + 0.0j

        for ell in range(nlay - 2, -1, -1):
            kz[ell] = np.sqrt(omega2 * (inv_vp2[ell] - p2) + 0j)
            kz[ell] = np.where(np.imag(kz[ell]) < 0, -kz[ell], kz[ell])
            Z[ell] = omega * rho[ell] / kz[ell]

            r = (Z[ell + 1] - Z[ell]) / (Z[ell + 1] + Z[ell])
            phase = np.exp(2.0j * kz[ell + 1] * h[ell + 1])
            D = 1.0 + r * Rstep[ell + 1] * phase

            Rstep[ell] = (r + Rstep[ell + 1] * phase) / D
        if free_surface:
            cavity = 1.0 / (1.0 + Rstep[0] * np.exp(2.0j * kz[0] * h[0]))
            ghost = -4.0 * np.sin(kz[0] * zs) * np.sin(kz[0] * zr)
            Rval = cavity * Rstep[0] * ghost
        else:
            Rval = Rstep[0]

        R[iw] = Rval

    return R

def reflectivity_q(layers, omegas, p, **kwargs):
    omegas = np.asarray(omegas, dtype=np.complex128)
    p = np.asarray(p, dtype=np.float64)
    R = numpy_reflectivity_p(layers, omegas, p, **kwargs)
    return R

# --- helper to call Fortran reflectivity ---
def fortran_reflectivity(layers, omegas, p, free_surface: bool = True, zr=70.0, zs=80.0):
    if not FORTRAN_AVAILABLE:
        raise RuntimeError("Fortran module not compiled/importable")
    nw = omegas.size
    nq = p.size

    h, vp, rho = map(
        lambda x: np.asfortranarray(x, dtype=np.float64),
        zip(*layers)
    )
    omegas = np.asfortranarray(omegas, dtype=np.complex128)
    p = np.asfortranarray(p, dtype=np.float64)
    #print("Calling Fortran reflectivity module...")
    #print(f"Layers: {h.size}, Frequencies: {nw}, q points: {nq}")
    #t0 = time.time()
    R = rfmod.compute_reflectivity(h, vp, rho, omegas, p, free_surface, zr, zs)
    #t1 = time.time()
    #print("fortran elapsed: {:.3f}s, R shape {}".format(t1-t0, R.shape))
    return R

def benchmark():
    # Example model (small for quick testing; scale up to test perf)
    layers = [
        (100.0, 1500.0, 1800.0),
        (250.0, 1900.0, 2000.0),
        (350.0, 3900.0, 2000.0),
        (400.0, 2900.0, 2000.0),
    ]
    freqs = np.linspace(0.1, 50.0, 1024, dtype=np.complex128)
    omegas = 2.0 * np.pi * freqs + 0.5 * 1j
    Ntheta = 6200
    thetas = np.linspace(0., np.pi, Ntheta, dtype=np.float64)
    p = np.sin(thetas) / layers[0][1]

    # warm-up
    print("warming up numpy implementation ...")
    #R_np = np.zeros((omegas.size, thetas.size), dtype=np.complex128)
    R_np = reflectivity_q(layers, omegas, p, free_surface=1, zr=70., zs=80.)
    if FORTRAN_AVAILABLE:
        print("warming up fortran implementation ...")
        #R_f = np.zeros((omegas.size, thetas.size), dtype=np.complex128)
        R_f = fortran_reflectivity(layers, omegas, p, free_surface=1, zr=70., zs=80.)

    # real benchmark
    print("\n ----- Benchmark ----- ")
    repeats = 4
    t_np = 0.0
    for r in range(repeats):
        t0 = time.time()
        R_np = reflectivity_q(layers, omegas, p, free_surface=1, zr=70., zs=80.)
        t1 = time.time()
        dt = t1 - t0
        print(f"numpy run {r+1}/{repeats} : {dt:.3f}s")
        t_np += dt
    t_np /= repeats

    if FORTRAN_AVAILABLE:
        t_f = 0.0
        for r in range(repeats):
            t0 = time.time()
            R_f = fortran_reflectivity(layers, omegas, p, free_surface=1, zr=70., zs=80.)
            t1 = time.time()
            dt = t1 - t0
            print(f"fortran run {r+1}/{repeats} : {dt:.3f}s")
            t_f += dt
        t_f /= repeats

        print("\nAverage times (s): numpy {:.3f}  fortran {:.3f}  speedup {:.2f}x".format(t_np, t_f, t_np / t_f))
    else:
        print("\nFortran module not available — only numpy timings shown.")

    err = np.max(np.abs(R_np - R_f))
    print(f"Max abs error between numpy and fortran: {err:.3e}")
    plot_reflectivity(omegas, thetas, R_np, omega_c=150.0)
    plot_reflectivity(omegas, thetas, R_f, omega_c=150.0)

if __name__ == "__main__":
    benchmark()