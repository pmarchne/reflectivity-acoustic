import numpy as np
from src.forward import forward
from src.layers import create_layers_from_interfaces


def l2_misfit(dcal, dobs, std_noise=1.):
    """ L2 misfit between predicted and observed data """
    residual = dcal - dobs  # residual for all source/receiver pairs
    return 0.5 * np.sum(residual**2) / (std_noise**2)


def fd_gradient_vp(vp, rho, z_interfaces, config, dobs, std_noise, eps=1e-3):
    grad = np.zeros_like(vp)

    for i in range(1, len(vp)):
        vp_p = vp.copy()
        vp_m = vp.copy()

        vp_p[i] += eps
        vp_m[i] -= eps

        layers_p = create_layers_from_interfaces(z_interfaces, vp_p, rho)
        layers_m = create_layers_from_interfaces(z_interfaces, vp_m, rho)

        d_p = forward(layers_p, config)[0]
        d_m = forward(layers_m, config)[0]

        phi_p = l2_misfit(d_p[0], dobs, std_noise)
        phi_m = l2_misfit(d_m[0], dobs, std_noise)

        grad[i] = (phi_p - phi_m) / (2 * eps)
        
    return grad

