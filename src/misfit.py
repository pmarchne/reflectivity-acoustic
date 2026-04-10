import numpy as np
# from src.parameters import Parameters
# from src.acquisition import Acquisition
from src.forward import forward
from src.layers import create_layers_from_interfaces
# from src.utilities import adjoint_inverse_fft_signal


def l2_misfit(dcal, dobs, std_noise):
    """ L2 misfit between predicted and observed data """
    residual = dcal - dobs  # residual for all source/receiver pairs
    return 0.5 * np.sum(residual**2) #/ (std_noise**2)


def fd_gradient_vp(vp, rho, z_interfaces, config, dobs, std_noise, eps=1e-3):
    grad = np.zeros_like(vp)

    for i in range(1, len(vp)):
        vp_p = vp.copy()
        vp_m = vp.copy()

        vp_p[i] += eps
        vp_m[i] -= eps

        layers_p = create_layers_from_interfaces(z_interfaces, vp_p, rho)
        layers_m = create_layers_from_interfaces(z_interfaces, vp_m, rho)

        d_p, _ = forward(layers_p, config)
        d_m, _ = forward(layers_m, config)

        phi_p = l2_misfit(d_p[0], dobs, std_noise)
        phi_m = l2_misfit(d_m[0], dobs, std_noise)

        grad[i] = (phi_p - phi_m) / (2 * eps)
        
    return grad

'''def l2_misfit(dcal, dobs, hs, vp, rho, std_noise):
    """ L2 misfit between predicted and observed data """
    layers_new = create_layers(hs=hs, vps=vp, rhos=rho)
    dcal = forward(layers_new, param.acqui, param,
                    nq_prop=param.nq_prop,
                    free_surface=param.fs)
    
    residual = dcal - dobs  # residual for all source/receiver pairs
    return 0.5 * np.sum(residual**2) / (std_noise**2)'''
