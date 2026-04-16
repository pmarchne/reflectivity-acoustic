import numpy as np
from src.fortran.reflectivity_benchmark import numpy_reflectivity_p
from src.utilities import timer

try:
    import src.fortran.reflectivity_adj as reflectivity_adj

    FORTRAN_AVAILABLE = True
    rfadjmod = reflectivity_adj.reflectivity_adj_mod
    # print(rfadjmod.compute_reflectivity_adj.__doc__)
except Exception as e:
    print("Fortran module not available:", e)
    FORTRAN_AVAILABLE = False
    rfadjmod = None


def numpy_reflectivity_p_adj(
    layers,
    omegas,
    p,
    free_surface=1,
    zr=70.0,
    zs=80.0,
):
    h, vp, rho = map(lambda x: np.asarray(x, dtype=np.complex128), zip(*layers))
    omegas = np.asarray(omegas, dtype=np.complex128)
    p = np.asarray(p, dtype=np.float64)
    p2 = p * p

    nlay = len(h)
    nw = omegas.size
    nk = p.size

    R = np.zeros((nw, nk), dtype=np.complex128)
    dR_dvp = np.zeros((nw, nk, nlay), dtype=np.complex128)
    dR_drho = np.zeros((nw, nk, nlay), dtype=np.complex128)

    inv_vp2 = 1.0 / (vp * vp)
    inv_vp3 = 1.0 / (vp * vp * vp)

    for iw, omega in enumerate(omegas):
        omega2 = omega * omega

        # Forward tape, store intermediate variables for adjoint
        kz = np.zeros((nlay, nk), dtype=np.complex128)
        Z = np.zeros((nlay, nk), dtype=np.complex128)
        Rstep = np.zeros((nlay, nk), dtype=np.complex128)
        Rval = np.zeros(nk, dtype=np.complex128)

        # Bottom layer
        kz[-1] = np.sqrt(omega2 * (inv_vp2[-1] - p2) + 0j)
        kz[-1] = np.where(np.imag(kz[-1]) < 0, -kz[-1], kz[-1])
        Z[-1] = omega * rho[-1] / kz[-1]
        Rstep[-1] = 0.0 + 0.0j

        # Upward recursion
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

        # Reverse tape
        adj_kz = np.zeros((nlay, nk), dtype=np.complex128)
        adj_Z = np.zeros((nlay, nk), dtype=np.complex128)

        if free_surface:
            adj_R = np.ones(nk, dtype=np.complex128)

            adj_cavity = adj_R * Rstep[0] * ghost
            adj_Rstep0 = adj_R * cavity * ghost
            adj_ghost = adj_R * cavity * Rstep[0]

            t = 1.0 + Rstep[0] * np.exp(2.0j * kz[0] * h[0])
            adj_t = -adj_cavity / (t * t)

            adj_Rstep0 += adj_t * np.exp(2.0j * kz[0] * h[0])
            adj_s0 = adj_t * Rstep[0]

            dghost_dkz = -4.0 * (
                np.cos(kz[0] * zs) * zs * np.sin(kz[0] * zr)
                + np.sin(kz[0] * zs) * np.cos(kz[0] * zr) * zr
            )

            adj_kz[0] += (
                adj_s0 * (2.0j * h[0] * np.exp(2.0j * kz[0] * h[0]))
                + adj_ghost * dghost_dkz
            )
            adj_current = adj_Rstep0
        else:
            adj_current = np.ones(nk, dtype=np.complex128)

        # Backward recursion over interfaces
        for ell in range(nlay - 1):
            x = Rstep[ell + 1]
            r = (Z[ell + 1] - Z[ell]) / (Z[ell + 1] + Z[ell])
            q = np.exp(2j * kz[ell + 1] * h[ell + 1])
            D = 1.0 + r * x * q

            # Local reverse derivatives of R = (r + x q) / (1 + r x q)
            adj_r = adj_current * (1.0 - (x * q) ** 2) / (D * D)
            adj_x = adj_current * q * (1.0 - r * r) / (D * D)
            adj_q = adj_current * x * (1.0 - r * r) / (D * D)

            B = Z[ell + 1] + Z[ell]
            invB2 = 1.0 / (B * B)

            # r = (Z_next - Z_cur) / (Z_next + Z_cur)
            adj_Z[ell + 1] += adj_r * (2.0 * Z[ell]) * invB2
            adj_Z[ell] += adj_r * (-2.0 * Z[ell + 1]) * invB2
            # q = exp(2 i kz_next h_{ell+1})
            adj_kz[ell + 1] += adj_q * (2.0j * h[ell + 1] * q)
            # carry adjoint to the next deeper Rstep
            adj_current = adj_x

        # Convert layer adjoints to parameter derivatives
        for ell in range(nlay):
            dkz_dvp = -omega2 * inv_vp3[ell] / kz[ell]
            dZ_dvp = -Z[ell] / kz[ell] * dkz_dvp
            dZ_drho = omega / kz[ell]

            dR_dvp[iw, :, ell] = adj_kz[ell] * dkz_dvp + adj_Z[ell] * dZ_dvp
            dR_drho[iw, :, ell] = adj_Z[ell] * dZ_drho

    return R, dR_dvp, dR_drho


def reflectivity_p_adj(layers, omegas, p, **kwargs):
    omegas = np.asarray(omegas, dtype=np.complex128)
    p = np.asarray(p, dtype=np.float64)
    R, dR_dvp, dR_drho = numpy_reflectivity_p_adj(layers, omegas, p, **kwargs)
    return R, dR_dvp, dR_drho


def gradient_check(layers, omegas, p, eps=1e-8, print_info=False, **kwargs):
    layers = np.asarray(layers, dtype=np.float64)
    z = layers[:, 0]
    vp0 = layers[:, 1].copy()
    rho0 = layers[:, 2].copy()

    R, dR_dvp, dR_drho = numpy_reflectivity_p_adj(layers, omegas, p, **kwargs)
    max_err_vp, max_err_rho = np.zeros_like(vp0), np.zeros_like(rho0)
    for ell in range(len(vp0)):
        vp_cs = vp0.astype(np.complex128)
        vp_cs[ell] += 1j * eps
        layers_cs = np.column_stack([z, vp_cs, rho0])
        R_cs = numpy_reflectivity_p(layers_cs, omegas, p, **kwargs)
        cs_vp = (R_cs - R) / (1j * eps)
        max_err_vp[ell] = np.max(np.abs(cs_vp - dR_dvp[:, :, ell]))

        rho_cs = rho0.astype(np.complex128)
        rho_cs[ell] += 1j * eps
        layers_cs = np.column_stack([z, vp0, rho_cs])
        R_cs = numpy_reflectivity_p(layers_cs, omegas, p, **kwargs)
        cs_rho = (R_cs - R) / (1j * eps)
        max_err_rho[ell] = np.max(np.abs(cs_rho - dR_drho[:, :, ell]))

        if print_info:
            print(
                f"layer {ell:d} | "
                f"vp max absolute err = {max_err_vp[ell]:.3e} | "
                f"rho max absolute err = {max_err_rho[ell]:.3e}"
            )
    return max_err_vp, max_err_rho


def fortran_reflectivity_adj(layers, omegas, p, free_surface=1, zr=70.0, zs=80.0):
    if not FORTRAN_AVAILABLE:
        raise RuntimeError("Fortran module not compiled/importable")
    h, vp, rho = map(lambda x: np.asfortranarray(x, dtype=np.float64), zip(*layers))
    omegas = np.asfortranarray(omegas, dtype=np.complex128)
    p = np.asfortranarray(p, dtype=np.float64)
    R, dR_dvp, dR_drho = rfadjmod.compute_reflectivity_adj(
        h, vp, rho, omegas, p, free_surface, zr, zs
    )
    return R, dR_dvp, dR_drho


def benchmark_adj():
    layers = [
        (100.0, 1500.0, 1800.0),
        (250.0, 1900.0, 2000.0),
        (300.0, 3900.0, 2500.0),
        (400.0, 2900.0, 2750.0),
    ]

    freqs = np.linspace(0.1, 50.0, 1024, dtype=np.float64)
    omegas = 2.0 * np.pi * freqs + 0.5 * 1j
    thetas = np.linspace(0.01, 0.99 * np.pi, 1200, dtype=np.float64)
    p = np.sin(thetas) / layers[0][1]

    fs, zs, zr = 1, 80.0, 70.0

    # warm-up
    print("warming up numpy implementation ...")
    tmp, tmp2, tmp3 = numpy_reflectivity_p_adj(
        layers, omegas, p, free_surface=fs, zr=zr, zs=zs
    )
    if FORTRAN_AVAILABLE:
        print("warming up fortran implementation ...")
        tmpf, tmpf2, tmpf3 = fortran_reflectivity_adj(
            layers, omegas, p, free_surface=fs, zr=zr, zs=zs
        )

    repeats = 5
    for _ in range(repeats):
        with timer("numpy reflectivity adjoint"):
            R, dR_dvp, dR_drho = numpy_reflectivity_p_adj(
                layers, omegas, p, free_surface=fs, zr=zr, zs=zs
            )
        if FORTRAN_AVAILABLE:
            with timer("fortran reflectivity adjoint"):
                Rf, dRf_dvp, dRf_drho = fortran_reflectivity_adj(
                    layers, omegas, p, free_surface=fs, zr=zr, zs=zs
                )
        err_fortran = np.max(np.abs(Rf - R))
        err_fortran_dvp = np.max(np.abs(dRf_dvp - dR_dvp))
        err_fortran_drho = np.max(np.abs(dRf_drho - dR_drho))
        print(f"\nMax abs err Fortran-numpy: {err_fortran:.3e}")
        print(f"Max abs err dR/dvp Fortran-numpy: {err_fortran_dvp:.3e}")
        print(f"Max abs err dR/drho Fortran-numpy: {err_fortran_drho:.3e}")

    print("\nPerforming gradient check with complex step ...")
    print("R shape      :", R.shape)
    print("dR_dvp shape  :", dR_dvp.shape)
    print("dR_drho shape :", dR_drho.shape)

    print("\n dR/dvp derivative value, per layer:", dR_dvp[50, 10, :])
    print("\n dR/drho derivative value, per layer:", dR_drho[50, 10, :])

    max_err_vp, max_err_rho = gradient_check(
        layers, omegas, p, eps=1e-8, print_info=True, free_surface=fs, zr=zr, zs=zs
    )
    print(max_err_rho)
    print(max_err_vp)


if __name__ == "__main__":
    benchmark_adj()
