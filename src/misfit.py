import numpy as np

from src.simulation import Simulation
from src.layers import create_layers


def l2_misfit(dcal, dobs, std_noise=1.0):
    """L2 misfit between predicted and observed data."""
    residual = dcal - dobs
    return 0.5 * np.sum(residual**2) / (std_noise**2)


def fd_gradient_vp(vp, rho, h, sim: Simulation, dobs, std_noise, eps=1e-3):
    """Finite-difference gradient of the L2 misfit with respect to vp."""
    grad = np.zeros_like(vp)

    for i in range(1, len(vp)):
        vp_p = vp.copy()
        vp_m = vp.copy()

        vp_p[i] += eps
        vp_m[i] -= eps

        layers_p = create_layers(h, vp_p, rho)
        layers_m = create_layers(h, vp_m, rho)

        d_p = sim.forward(layers_p)
        d_m = sim.forward(layers_m)

        phi_p = l2_misfit(d_p[0], dobs, std_noise)
        phi_m = l2_misfit(d_m[0], dobs, std_noise)

        grad[i] = (phi_p - phi_m) / (2 * eps)

    return grad


def fd_gradient_rho(vp, rho, h, sim: Simulation, dobs, std_noise, eps=1e-3):
    """Finite-difference gradient of the L2 misfit with respect to rho."""
    grad = np.zeros_like(rho)

    for i in range(1, len(rho)):
        rho_p = rho.copy()
        rho_m = rho.copy()

        rho_p[i] += eps
        rho_m[i] -= eps

        # Create layers with perturbed rho
        layers_p = create_layers(h, vp, rho_p)
        layers_m = create_layers(h, vp, rho_m)

        # Forward simulations
        d_p = sim.forward(layers_p)
        d_m = sim.forward(layers_m)

        # Misfit calculation
        phi_p = l2_misfit(d_p[0], dobs, std_noise)
        phi_m = l2_misfit(d_m[0], dobs, std_noise)

        grad[i] = (phi_p - phi_m) / (2 * eps)

    return grad


def fd_gradient_h(vp, rho, h, sim: Simulation, dobs, std_noise, eps=1e-3):
    """Finite-difference gradient of the L2 misfit with respect to h."""
    grad = np.zeros_like(h)

    for i in range(1, len(h)):
        h_p = h.copy()
        h_m = h.copy()

        h_p[i] += eps
        h_m[i] -= eps

        # Create layers with perturbed h
        layers_p = create_layers(h_p, vp, rho)
        layers_m = create_layers(h_m, vp, rho)

        # Forward simulations
        d_p = sim.forward(layers_p)
        d_m = sim.forward(layers_m)

        # Misfit calculation
        phi_p = l2_misfit(d_p[0], dobs, std_noise)
        phi_m = l2_misfit(d_m[0], dobs, std_noise)

        grad[i] = (phi_p - phi_m) / (2 * eps)

    return grad