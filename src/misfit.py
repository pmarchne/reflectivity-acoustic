import numpy as np
from parameters import Parameters
from acquisition import Acquisition
from forward import forward

def l2_misfit(dcal, dobs, layers, acqui: Acquisition, param: Parameters, std_noise=1.):
    """ L2 misfit between predicted and observed data """
    nobs = float(dobs.shape[1])
    dcal = forward(layers, acqui, param)
    residual = dcal - dobs
    return 0.5 * param.dt * np.sum(residual**2) / (std_noise**2*nobs)

def grad_l2_misft(d_cal, d_obs, layers, acqui: Acquisition, param: Parameters):
    "to be implemented ..."
    return None