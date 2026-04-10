import numpy as np
from src.config import Config
from src.builders import build_problem
from src.layers import create_layers_from_interfaces
from src.forward import forward
from src.noise import add_noise
from src.plot.plot_tools import plot_seismogram
from src.misfit import l2_misfit, fd_gradient_vp
from src.adjoint import compute_gradient
from src.utilities import source_frequency, timer

# -----------------------
# Build experiment
# -----------------------
config = Config(n_receivers=128, noise_level=1.,
                x_min=0., x_max=700.,
                z_rec=75., z_src=50., nq_prop=1024, nq_evan=256, f0=10.,
                total_time=1.024, delay=0.2, 
                source_deriv=True, epsilon=1.5, 
                free_surface=True)
param, acq = build_problem(config)
print(config)

# -----------------------
# model
# -----------------------
z_interfaces = np.array([0.0, 100.0, 200.0, 250.0, 350.0, 450.0,
                         550.0, 650.0, 700.0])
vp = np.array([1505.0, 1603.0, 1749.0, 2019.0, 2179.0, 1900.0, 2265.0, 3281.0])
rho = np.full_like(vp, 2000.0)
layers = create_layers_from_interfaces(z_interfaces, vp, rho)

# -----------------------
# Observations
# -----------------------
d_clean, _ = forward(layers, config, timing=True)
#d_obs, std_noise = add_noise(d_clean, config.noise_level, seed=config.seed)

d_cal_seis = d_clean[0, :, :]
d_cal_seis = d_cal_seis / np.max(np.abs(d_cal_seis))
plot_seismogram(d_cal_seis.T, acq.xr, param.time, vmin=-1, vmax=1,
                cmap='gray_r', ncolors=256, figsize=(7, 7))


# -----------------------
# Change vp and get gradient
# -----------------------
#vp_new = np.array([1505.0, 1603.0, 1749.0, 2019.0, 2179.0, 1900.0, 2265.0, 3281.0])
vp_new = np.array([1505.0, 1643.0, 2749.0, 2219.0, 3400.0, 2900.0, 2065.0, 4281.0])
layers_new = create_layers_from_interfaces(z_interfaces, vp_new, rho)
d_new, cache = forward(layers_new, config, timing=True)
residual = d_new - d_clean
residual = residual[0, :, :]
plot_seismogram(residual.T, acq.xr, param.time,
                cmap='gray_r', ncolors=256, figsize=(7, 7))

tmp = d_clean[0] - d_new[0]
print("tmp shape", tmp.shape)

phi0 = l2_misfit(d_clean[0], d_new[0], 1.)
print("l2 misfit", phi0)

source_freq, omegas = source_frequency(param, config)

with timer("grad adj 1"):
    grad_vp, grad_rho = compute_gradient(
        residual, layers_new, omegas, 
        source_freq, config, cache
    )

with timer("grad adj 2"):
    grad_vp, grad_rho = compute_gradient(
        residual, layers_new, omegas, 
        source_freq, config, cache
    )
print("gradient vp adj", grad_vp)
print("gradient rho adj", grad_rho)

with timer("fd grad"):
  grad_vp_fd = fd_gradient_vp(
      vp_new,
      rho,
      z_interfaces,
      config,
      d_clean[0],
      1.
  )

print("fd gradient vp", grad_vp_fd)
print("------------------")

layers = create_layers_from_interfaces(z_interfaces, vp, rho)
d0, _ = forward(layers, config)

def J_forward(dm, eps=1e-6):
    vp_perturbed = vp + eps * dm
    layers_p = create_layers_from_interfaces(z_interfaces, vp_perturbed, rho)

    d_p, _ = forward(layers_p, config)

    return (d_p[0] - d0[0]) / eps

def J_adjoint(r):
    grad_vp, _ = compute_gradient(
        residual=r,
        layers=layers,
        omegas=param.omegas,
        source_freq=source_freq,
        config=config,
        cache=cache,
    )
    return grad_vp


rng = np.random.default_rng(0)

for _ in range(12):
    dm = rng.standard_normal(vp.shape)
    dm[0] = 0.0
    r = rng.standard_normal(d0[0].shape)

    Jdm = J_forward(dm)
    Jtr = J_adjoint(r)

    lhs = np.vdot(Jdm.ravel(), r.ravel())
    rhs = np.vdot(dm[1:].ravel(), Jtr[1:].ravel())

    rel_err = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-16)
    #print("lhs:", lhs)
    #print("rhs:", rhs)
    print("rel_err:", rel_err)

