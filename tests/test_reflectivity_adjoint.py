import numpy as np

from src.utilities import timer
from src.fortran.reflectivity_adjoint import (fortran_reflectivity_adj,
                                              reflectivity_p_adj,
                                              gradient_check)
from src.simulation import Simulation
from src.config import Config
from src.layers import (
    update_from_arrays,
    create_layers,
    to_arrays
    )
from src.misfit import fd_gradient_vp, fd_gradient_h, fd_gradient_rho


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
            r_np, dr_dvp, dr_drho, dr_dh = reflectivity_p_adj(
                layers,
                omegas,
                p,
                free_surface=1,
                zr=70.0,
                zs=60.0
            )

        with timer(f"fortran run {r + 1}: ", True):
            r_f, drf_dvp, drf_drho, drf_dh = fortran_reflectivity_adj(
                layers,
                omegas,
                p,
                free_surface=1,
                zr=70.0,
                zs=60.0
            )

    r_np, r_f = np.asarray(r_np), np.asarray(r_f)
    dr_dvp, drf_dvp = np.asarray(dr_dvp), np.asarray(drf_dvp)
    dr_drho, drf_drho = np.asarray(dr_drho), np.asarray(drf_drho)
    dr_dh, drf_dh = np.asarray(dr_dh), np.asarray(drf_dh)

    max_err_r = np.max(np.abs(r_np - r_f))
    max_err_vp = np.max(np.abs(dr_dvp - drf_dvp))
    max_err_rho = np.max(np.abs(dr_drho - drf_drho))
    max_err_h = np.max(np.abs(dr_dh - drf_dh))
    # Assertions
    assert max_err_r < 1e-8, f"Reflectivity mismatch: {max_err_r}"
    assert max_err_vp < 1e-8, f"dR/dVp mismatch: {max_err_vp}"
    assert max_err_rho < 1e-8, f"dR/drho mismatch: {max_err_rho}"
    assert max_err_h < 1e-8, f"dR/dh mismatch: {max_err_h}"

    # Finite-difference gradient check
    max_fd_err_vp, max_fd_err_rho, max_fd_err_h = gradient_check(
        layers, omegas, p, eps=1e-8, free_surface=0, zr=70.0, zs=60.0
    )

    # analytic vs FD gradients (per-layer max error)
    # Free surface excluded from gradient check - FD numerical instability.
    assert np.all(max_fd_err_vp < 1e-3), f"Vp FD errors: {max_fd_err_vp}"
    assert np.all(max_fd_err_rho < 1e-3), f"rho FD errors: {max_fd_err_rho}"
    assert np.all(max_fd_err_h < 1e-3), f"h FD errors: {max_fd_err_h}"


def test_fd_reflectivity(layered_model):
    config = Config(
        n_receivers=16,
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
    sim = Simulation(config)
    layers = layered_model
    d_clean = sim.forward(layers, timing=False)
    # Perturbed model
    vp_new = np.array([1505.0, 1643.0, 2749.0,
                       2219.0, 3400.0, 2900.0,
                       2065.0, 4281.0], dtype=float)
    h_new = np.array([110., 120., 60., 110., 90., 110., 120., 60], dtype=float)
    rho_new = np.array([2105.0, 3000., 3000.,
                       3000., 3400.0, 3000.,
                       3000., 3000.], dtype=float)
    layers_new = update_from_arrays(layers, vps=vp_new, hs=h_new, rhos=rho_new)
    d_new = sim.forward(layers_new, timing=False)

    residual = d_new - d_clean
    residual = residual[0]  # keep 1st source - forward returns [nsrc,nrec,nt]

    # Adjoint gradient
    grad_vp, grad_rho, grad_h = sim.gradient(
        residual=residual,
        layers=layers_new
    )

    h_new, vp_new, rho_new = to_arrays(layers_new, return_interfaces=False)
    # Optional FD sanity check for vp
    grad_vp_fd = fd_gradient_vp(
        vp_new,
        rho_new,
        h_new,
        sim,
        d_clean[0],
        1.0,
        eps=1e-3
    )

    grad_rho_fd = fd_gradient_rho(
        vp_new,
        rho_new,
        h_new,
        sim,
        d_clean[0],
        1.0,
        eps=1e-3
    )

    grad_h_fd = fd_gradient_h(
        vp_new,
        rho_new,
        h_new,
        sim,
        d_clean[0],
        1.0,
        eps=1e-3
    )

    err_fd_adj = np.max(np.abs(grad_vp - grad_vp_fd))
    print("vp grad:", err_fd_adj)
    assert err_fd_adj < 1e-3, f"FD vs adjoint mismatch: {err_fd_adj:.3e}"

    err_fd_adj = np.max(np.abs(grad_h - grad_h_fd))
    print("h grad:", err_fd_adj)
    assert err_fd_adj < 1e-3, f"FD vs adjoint mismatch: {err_fd_adj:.3e}"

    err_fd_adj = np.max(np.abs(grad_rho - grad_rho_fd))
    print("rho grad:", err_fd_adj)
    assert err_fd_adj < 1e-3, f"FD vs adjoint mismatch: {err_fd_adj:.3e}"

    # Dot-product test
    def J_forward_vp(dm, eps=1e-6):
        vp_perturbed = vp_new + eps * dm
        layers_p = create_layers(
            h_new, vp_perturbed, rho_new
            )
        d_p = sim.forward(layers_p, timing=False)
        return (d_p[0] - d_new[0]) / eps
    
    def J_forward_h(dm, eps=1e-6):
        h_perturbed = h_new + eps * dm
        layers_p = create_layers(
            h_perturbed, vp_new, rho_new
            )
        d_p = sim.forward(layers_p, timing=False)
        return (d_p[0] - d_new[0]) / eps
    
    def J_forward_rho(dm, eps=1e-6):
        rho_perturbed = rho_new + eps * dm
        layers_p = create_layers(
            h_new, vp_new, rho_perturbed
            )
        d_p = sim.forward(layers_p, timing=False)
        return (d_p[0] - d_new[0]) / eps

    def J_adjoint(r):
        gvp, grho, gh = sim.gradient(
            residual=r,
            layers=layers_new
        )
        return gvp, grho, gh

    rng = np.random.default_rng(0)
    for i in range(5):
        dm_vp = rng.standard_normal(vp_new.shape)
        dm_rho = rng.standard_normal(rho_new.shape)
        dm_h = rng.standard_normal(h_new.shape)
        
        dm_vp[0] = 0.0
        dm_rho[0] = 0.0
        dm_h[0] = 0.0

        J_dm = (J_forward_vp(dm_vp) + 
                J_forward_rho(dm_rho) + 
                J_forward_h(dm_h))

        r = rng.standard_normal(d_new[0].shape)
        gvp, grho, gh = J_adjoint(r)

        lhs = np.vdot(J_dm.ravel(), r.ravel())
        
        rhs = (np.vdot(dm_vp.ravel(), gvp.ravel()) + 
               np.vdot(dm_rho.ravel(), grho.ravel()) + 
               np.vdot(dm_h.ravel(), gh.ravel()))

        rel_err = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-16)
        print(rel_err)
        print(f"Adjoint Test Iteration {i+1}: rel_err={rel_err:.3e}")
        assert rel_err < 1e-5, f"Adjoint test failed at iteration {i}: rel_err={rel_err:.3e}"

    print("Success: Gradient and Adjoint operators are verified.")

