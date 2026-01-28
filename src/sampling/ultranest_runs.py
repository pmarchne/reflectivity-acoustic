import os
import sys
sys.path.append(os.path.abspath(os.path.join("../../")))

import numpy as np
from src.config import FWIConfig, build_problem
from src.layers import create_layers
from src.forward import forward
from src.misfit import add_noise
from src.sampling.model import FWILogPosterior
import ultranest, pickle
from src.plot.plot_tools import plot_seismogram
config = FWIConfig()
param, acq = build_problem(config)

vps_ref = np.array([1500., 1900., 2800., 3800., 2300., 5000.])
hs = np.array([150., 200., 250., 380., 440., 500.])
rhos = np.full_like(vps_ref, 2000.)
layers = create_layers(hs, vps_ref, rhos)

d_clean = forward(layers, acq, param, nq_prop=config.nq_prop, free_surface=config.fs)
d_obs, std_noise = add_noise(d_clean, config.noise_level)
plot_seismogram(d_obs[0, :, :], acq.xr, param.time, vmin=-0.06, vmax=0.06)

prior_mean = np.array([2000., 2500., 2500., 3000., 3000.])
sigma = np.array([1000., 1500., 1500., 1500., 1500.])
prior_cov = np.diag(sigma**2)

model = FWILogPosterior(
    dobs=d_obs,
    layer=layers,
    param=param,
    acqui=acq,
    config=config,
    noise_std=std_noise,
    prior_mean=prior_mean,
    prior_cov=prior_cov
)

def ultranest_prior(cube):
    params = cube.copy()
    # transform location parameter: uniform prior
    lo = 1000
    hi = 7000
    for i in range(len(prior_mean)):
        params[i] = cube[i] * (hi - lo) + lo
    return params

def ultranest_likelihood_weak(params):
    c1, c2, c3, c4, c5 = params
    c = np.array([c1, c2, c3, c4, c5])
    return model(c)/16.

def ultranest_likelihood(params):
    c1, c2, c3, c4, c5 = params
    c = np.array([c1, c2, c3, c4, c5])
    return model(c)

def ultranest_likelihood_strong(params):
    c1, c2, c3, c4, c5 = params
    c = np.array([c1, c2, c3, c4, c5])
    return 16.*model(c)

vp0 = prior_mean.copy()
print("log_prior(mu):", model.log_prior(vp0))
print("log_likelihood(mu):", model.log_likelihood(vp0))

vp_test = vp0 + sigma   # one standard deviation perturbation
print("log_prior(mu + sigma):", model.log_prior(vp_test))
print("log_likelihood(mu + sigma):", model.log_likelihood(vp_test))

print("log_likelihood(ref):", model.log_likelihood(vps_ref[1:]))

param_names = ['Vp^2', 'Vp^3', 'Vp^4', 'Vp^5', 'Vp^6']
sampler = ultranest.ReactiveNestedSampler(param_names, 
                                          ultranest_likelihood, 
                                          ultranest_prior)

result = sampler.run(min_ess=500, 
                     min_num_live_points=1500, frac_remain=1e-2, 
                     max_ncalls=500000)
sampler.print_results()

filename = 'results_ultranest'
with open(filename + '.pkl', 'wb') as fp:
    pickle.dump(result, fp)
    print("results saved !")

