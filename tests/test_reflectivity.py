from pathlib import Path

import numpy as np
import pytest

from src.utilities import timer
from src.fortran.reflectivity_benchmark import reflectivity, fortran_reflectivity
from src.simulation import Simulation


def test_reflectivity_benchmark():
    layers = [
        (100.0, 1500.0, 1800.0),
        (250.0, 1900.0, 2000.0),
        (350.0, 3900.0, 2000.0),
        (400.0, 2900.0, 2000.0),
    ]

    freqs = np.linspace(0.1, 50.0, 1024, dtype=np.complex128)
    omegas = 2.0 * np.pi * freqs + 0.5j
    thetas = np.linspace(0.0, np.pi, 2000, dtype=np.float64)
    p = np.sin(thetas) / layers[0][1]

    repeats = 5
    for r in range(repeats):
        with timer(f"numpy run {r + 1}: ", True):
            r_np = reflectivity(layers, omegas, p,
                                  free_surface=1,
                                  zr=70.0, zs=80.0)
        with timer(f"fortran run {r + 1}: ", True):
            r_f = fortran_reflectivity(layers, omegas, p,
                                       free_surface=1,
                                       zr=70.0, zs=80.0)

    r_np = np.asarray(r_np)
    r_f = np.asarray(r_f)
    max_err = np.max(np.abs(r_np - r_f))
    assert max_err < 1e-8


def test_fd_reflectivity(param_fd, config_fd, layered_model):
    base_dir = Path(__file__).resolve().parents[1]
    file_path = base_dir / "FD_comparison" / "data" / "seis_v1_fs"
    if not file_path.exists():
        pytest.skip(f"Missing reference file: {file_path}")

    seismo_fd = np.fromfile(file_path, dtype=np.float32)

    nt_ref = 2048
    nr = config_fd.n_receivers
    seismo_fd = seismo_fd.reshape((nr, nt_ref)).T
    seismo_fd = seismo_fd / np.max(np.abs(seismo_fd))

    time_ref = np.linspace(0.0, config_fd.total_time, nt_ref)

    sim = Simulation(config_fd)

    # warm-up
    sim.forward(layered_model, timing=False)
    d_cal, _ = sim.forward(layered_model, timing=True)

    # expected shape: (Ns, Nr, Nt)
    ref = d_cal[0]
    ref = ref / np.max(np.abs(ref))

    mask = time_ref > 0.6
    err_max = np.zeros(nr)
    err_l2 = np.zeros(nr)

    for i in range(nr):
        trace_ref = ref[i, :]
        trace_ref_interp = np.interp(time_ref, param_fd.time, trace_ref)
        trace_fd = seismo_fd[:, i]
        residual = trace_ref_interp - trace_fd

        err_max[i] = np.max(np.abs(residual))
        err_l2[i] = np.sqrt(np.sum(residual[mask] ** 2))

    print("Err max reflectivity - FD", np.max(err_max))
    print("Err L2 reflectivity - FD", np.max(err_l2))
    assert np.max(err_max) < 5e-2
    assert np.max(err_l2) < 1e-1
