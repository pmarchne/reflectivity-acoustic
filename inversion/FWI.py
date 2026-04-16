import numpy as np
from scipy.optimize import minimize
from src.config import Config
from src.layers import create_layers_from_interfaces, update_layer_slice
from src.simulation import Simulation
from src.misfit import l2_misfit


def make_fwi_objective(d_obs, layers, sim, cost_history):
    """Define FWI objective function """
    def fwi_objective(vp_vec):
        # Build model
        lay = update_layer_slice(layers, vp_slice=vp_vec, start=1)
        # Forward
        d_cal, cache = sim.forward(lay, timing=True)
        # Misfit
        residual = d_cal - d_obs
        phi = l2_misfit(d_cal, d_obs)
        print("current phi", phi)
        # Gradient via adjoint
        grad_vp, _ = sim.gradient(
            residual=residual[0],
            layers=lay,
            cache=cache,
        )
        grad_opt = grad_vp[1:]  # first layer gradient is set to 0
        # Store cost
        cost_history.append(phi)
        print(f"phi = {phi:.3e} | vp = {vp_vec}")
        return phi, grad_opt
    return fwi_objective


def FWI_scipy():
    # Build experiment
    config = Config(n_receivers=16, noise_level=1.,
                    x_min=0., x_max=700.,
                    z_rec=75., z_src=50., nq_prop=512, nq_evan=128, f0=10.,
                    total_time=1.024, delay=0.2,
                    source_deriv=True, epsilon=1.5,
                    free_surface=True)
    sim = Simulation(config)

    # true model
    z_interfaces = np.array([0.0, 100.0, 250.0, 400.0, 700.0])
    # try this for cycle-skipping failure
    # vp_true = np.array([1500.0, 3000.0, 1800.0, 3500.0])
    vp_true = np.array([1500.0, 3000.0, 2800.0, 3500.0])
    rho = np.array([1200.0, 2100.0, 2000.0, 2200.0])
    layers = create_layers_from_interfaces(z_interfaces, vp_true, rho)

    # Observed data
    d_obs, _ = sim.forward(layers, timing=True)

    # initial model
    vp_init = np.array([3500.0, 3500.0, 3500.0])

    cost_history = []
    objective = make_fwi_objective(
        d_obs=d_obs,
        layers=layers,
        sim=sim,
        cost_history=cost_history,
    )
    # Run L-BFGS
    bounds = [(500.0, 7000.0)] * len(vp_init)
    result = minimize(
        objective,
        vp_init,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={
            "maxiter": 50,
            "disp": True,
        },
    )

    vp_est = result.x
    print("\nTrue vp :", vp_true)
    print("Init vp :", vp_init)
    print("Est vp  :", vp_est)
    print("\n")
    print("cost history", cost_history)


if __name__ == "__main__":
    FWI_scipy()
