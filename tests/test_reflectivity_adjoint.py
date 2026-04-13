import numpy as np

from src.utilities import timer
from src.fortran.reflectivity_adjoint import (fortran_reflectivity_adj,
                                              reflectivity_p_adj,
                                              gradient_check)
from src.forward import forward
from src.builders import build_problem
from src.config import Config
from src.layers import (
    update_from_arrays,
    create_layers_from_interfaces,
    to_arrays
    )
from src.utilities import source_frequency
from src.misfit import fd_gradient_vp
from src.adjoint import compute_gradient


def test_adj_reflectivity_benchmark():
    layers = [
        (100.0, 1500.0, 1800.0),
        (150.0, 2900.0, 2400.0),
        (220.0, 3900.0, 2500.0),
        (400.0, 2200.0, 3100.0),
        (550.0, 4200.0, 2000.0),
    ]

    freqs = np.linspace(0.01, 70.0, 1024, dtype=np.complex128)
    omegas = 2.0 * np.pi * freqs + 0.25j
    thetas = np.linspace(0.0, np.pi, 1200, dtype=np.float64)
    p = np.sin(thetas) / layers[0][1]

    repeats = 4
    for r in range(repeats):
        with timer(f"numpy run {r + 1}: ", True):
            r_np, dr_dvp, dr_drho = reflectivity_p_adj(
                layers,
                omegas,
                p,
                free_surface=1,
                zr=70.0,
                zs=60.0
            )

        with timer(f"fortran run {r + 1}: ", True):
            r_f, drf_dvp, drf_drho = fortran_reflectivity_adj(
                layers,
                omegas,
                p,
                free_surface=1,
                zr=70.0,
                zs=60.0
            )

    r_np = np.asarray(r_np)
    dr_dvp = np.asarray(dr_dvp)
    dr_drho = np.asarray(dr_drho)

    r_f = np.asarray(r_f)
    drf_dvp = np.asarray(drf_dvp)
    drf_drho = np.asarray(drf_drho)

    max_err_r = np.max(np.abs(r_np - r_f))
    max_err_vp = np.max(np.abs(dr_dvp - drf_dvp))
    max_err_rho = np.max(np.abs(dr_drho - drf_drho))
    # Assertions
    assert max_err_r < 1e-8, f"Reflectivity mismatch: {max_err_r}"
    assert max_err_vp < 1e-8, f"dR/dVp mismatch: {max_err_vp}"
    assert max_err_rho < 1e-8, f"dR/drho mismatch: {max_err_rho}"

    # Finite-difference gradient check
    max_fd_err_vp, max_fd_err_rho = gradient_check(
        layers,
        omegas,
        p,
        eps=1e-8,
        free_surface=0,
        zr=70.0,
        zs=60.0
    )

    # analytic vs FD gradients (per-layer max error)
    # Free surface excluded from gradient check - FD numerical instability.
    assert np.all(max_fd_err_vp < 1e-3), f"Vp FD errors: {max_fd_err_vp}"
    assert np.all(max_fd_err_rho < 1e-3), f"rho FD errors: {max_fd_err_rho}"


def test_fd_reflectivity(layered_model):
    config = Config(
        n_receivers=16,
        noise_level=1.0,
        x_min=0.0,
        x_max=700.0,
        z_rec=75.0,
        z_src=50.0,
        nq_prop=512,
        nq_evan=256,
        f0=10.0,
        total_time=1.024,
        delay=0.2,
        source_deriv=True,
        epsilon=1.5,
        free_surface=True,
    )
    param, _ = build_problem(config)
    layers = layered_model
    d_clean, _ = forward(layers, config, timing=False)
    # Perturbed model
    vp_new = np.array([1505.0, 1643.0, 2749.0,
                       2219.0, 3400.0, 2900.0,
                       2065.0, 4281.0], dtype=float)
    layers_new = update_from_arrays(layers, vps=vp_new)
    d_new, cache_new = forward(layers_new, config, timing=False)

    residual = d_new - d_clean
    residual = residual[0]  # keep 1st source - forward returns [nsrc,nrec,nt]
    source_freq = source_frequency(param, config)

    # Adjoint
    grad_vp, grad_rho = compute_gradient(
        residual=residual,
        layers=layers_new,
        source_freq=source_freq,
        config=config,
        cache=cache_new,
    )

    z_interfaces, vp_new, rho = to_arrays(layers_new, return_interfaces=True)
    # Optional FD sanity check for vp
    grad_vp_fd = fd_gradient_vp(
        vp_new,
        rho,
        z_interfaces,
        config,
        d_clean[0],
        1.0,
        eps=1e-3
    )

    err_fd_adj = np.max(np.abs(grad_vp - grad_vp_fd))
    print(err_fd_adj)
    assert err_fd_adj < 1e-3, f"FD vs adjoint mismatch: {err_fd_adj:.3e}"

    # Dot-product test
    def J_forward(dm, eps=1e-6):
        vp_perturbed = vp_new + eps * dm
        layers_p = create_layers_from_interfaces(
            z_interfaces, vp_perturbed, rho
            )
        d_p, _ = forward(layers_p, config, timing=False)
        return (d_p[0] - d_new[0]) / eps

    def J_adjoint(r):
        gvp, _ = compute_gradient(
            residual=r,
            layers=layers_new,
            source_freq=source_freq,
            config=config,
            cache=cache_new,
        )
        return gvp

    rng = np.random.default_rng(0)

    for _ in range(5):
        dm = rng.standard_normal(vp_new.shape)
        dm[0] = 0.0

        r = rng.standard_normal(d_new[0].shape)

        Jdm = J_forward(dm)
        Jtr = J_adjoint(r)

        lhs = np.vdot(Jdm.ravel(), r.ravel())
        rhs = np.vdot(dm[1:].ravel(), Jtr[1:].ravel())

        rel_err = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-16)
        print(rel_err)
        assert rel_err < 1e-6, f"Adjoint test failed: rel_err={rel_err:.3e}"
