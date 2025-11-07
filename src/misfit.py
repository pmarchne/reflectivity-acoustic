import numpy as np
from parameters import Parameters
from acquisition import Acquisition
from forward import forward

def l2_misfit(d_cal, d_obs, layers, acqui: Acquisition, param: Parameters):
    """ L2 misfit between predicted and observed data """
    #d_cal = forward(layers, acqui, param)
    residual = d_cal - d_obs
    return 0.5 * np.sum(residual**2)

def grad_l2_misft(d_cal, d_obs, layers, acqui: Acquisition, param: Parameters):
    "to be implemented ..."
    return None