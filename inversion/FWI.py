import numpy as np
from scipy.optimize import minimize
from src.config import Config
from src.layers import create_layers_from_interfaces, update_layer_slice
from src.simulation import Simulation
from src.misfit import l2_misfit


def make_fwi_objective(d_obs, layers, sim, cost_history):
    """Define FWI objective function"""

    def fwi_objective(vp_vec):
        # Build model
        lay = update_layer_slice(layers, vp_slice=vp_vec, start=1)
        # Forward
        d_cal, cache = sim.forward(lay)
        # Misfit
        residual = d_cal - d_obs
        phi = l2_misfit(d_cal, d_obs)

        # Gradient via adjoint
        grad_vp, _ = sim.gradient(
            residual=residual[0],
            layers=lay,
            cache=cache,
        )
        grad_opt = grad_vp[1:]  # first layer gradient is set to 0
        # Store cost
        cost_history.append(phi)
        vp_str = "[" + ", ".join(f"{v:7.1f}" for v in vp_vec) + "]"
        print(f"Iter {len(cost_history):>2} | misfit: {phi:.4e} | Vp: {vp_str}")

        return phi, grad_opt

    return fwi_objective


def fwi_scipy():
    # Build experiment
    config = Config(
        n_receivers=16,
        noise_level=1.0,
        x_min=0.0,
        x_max=700.0,
        z_rec=75.0,
        z_src=50.0,
        nq_prop=512,
        nq_evan=128,
        f0=10.0,
        total_time=1.024,
        delay=0.2,
        source_deriv=True,
        epsilon=1.5,
        free_surface=True,
    )
    sim = Simulation(config)

    # true model
    z_interfaces = np.array([0.0, 100.0, 250.0, 400.0, 700.0])
    # try this for cycle-skipping failure
    # vp_true = np.array([1500.0, 3000.0, 1800.0, 3500.0])
    vp_true = np.array([1500.0, 3000.0, 2800.0, 3500.0])
    print(f"Target Vp: {vp_true[1:]}")
    rho = np.array([1200.0, 2100.0, 2000.0, 2200.0])
    layers = create_layers_from_interfaces(z_interfaces, vp_true, rho)

    # Observed data
    d_obs, _ = sim.forward(layers)
    vp_init = np.array([3500.0, 4000.0, 4000.0])

    print(f"Starting FWI (L-BFGS-B)\n{'-'*30}")
    result = minimize(
        make_fwi_objective(d_obs, layers, sim, []),
        vp_init,
        method="L-BFGS-B",
        jac=True,
        bounds=[(500.0, 7000.0)] * len(vp_init),
        options={"maxiter": 50, "disp": False},
    )
    print(f"{'-'*50}\nInverted Vp: {result.x}")


if __name__ == "__main__":
    fwi_scipy()
