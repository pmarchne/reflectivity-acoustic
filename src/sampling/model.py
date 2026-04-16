import time
import numpy as np
from scipy.linalg import cho_factor, cho_solve

from src.layers import create_layers, to_arrays
from src.simulation import Simulation
from src.misfit import l2_misfit


class FWILogPosterior:
    def __init__(self, dobs, layer, config, noise_std, prior_mean, prior_cov):
        self.dobs = dobs
        self.layer = layer
        self.sim = Simulation(config)
        self.noise_std = noise_std

        self.mu = prior_mean
        self.cov = prior_cov
        self.chol = cho_factor(self.cov)
        self.inv_cov = cho_solve(self.chol, np.eye(len(self.mu)))

    @property
    def config(self):
        return self.sim.config

    # ---------------------
    # Likelihood
    # ---------------------
    def log_likelihood(self, vp):
        hs, _, rhos = to_arrays(self.layer)
        vp = np.insert(vp, 0, 1500.0)
        layers_new = create_layers(hs=hs, vps=vp, rhos=rhos)
        dcal, _ = self.sim.forward(layers_new)
        misfit = l2_misfit(dcal[0], self.dobs, std_noise=self.noise_std)
        misfit = misfit / self.sim.config.n_receivers
        return -misfit

    def grad_log_likelihood_fd(self, vp, eps=1e-3):
        grad = np.zeros_like(vp)
        for i in range(len(vp)):
            vp_p = vp.copy()
            vp_m = vp.copy()
            dv = eps * max(1.0, abs(vp[i]))
            vp_p[i] += dv
            vp_m[i] -= dv
            grad[i] = (self.log_likelihood(vp_p) - self.log_likelihood(vp_m)) / (2 * dv)
        return grad

    # ---------------------
    # Prior
    # ---------------------
    def log_prior(self, vp):
        diff = vp - self.mu
        return -0.5 * diff @ self.inv_cov @ diff

    def grad_log_prior(self, vp):
        return -self.inv_cov @ (vp - self.mu)

    # ---------------------
    # Posterior
    # ---------------------
    def __call__(self, vp):
        return self.log_prior(vp) + self.log_likelihood(vp)

    def grad(self, vp):
        return self.grad_log_prior(vp) + self.grad_log_likelihood_fd(vp)

    def cost_on_grid(self, ind1, ind2, v_ref, vmin=1000.0, vmax=6000.0, npts=80):
        x_vals = np.linspace(vmin, vmax, npts)
        y_vals = np.linspace(vmin, vmax, npts)
        xgrid, ygrid = np.meshgrid(x_vals, y_vals, indexing="ij")
        COST = np.empty_like(xgrid)
        hs, _, rhos = to_arrays(self.layer)

        start_time = time.time()
        for i in range(npts):
            for j in range(npts):
                VP = v_ref.copy()
                VP[ind1] = xgrid[i, j]
                VP[ind2] = ygrid[i, j]
                VP = np.insert(VP, 0, 1500.0)
                layers = create_layers(hs=hs, vps=VP, rhos=rhos)
                dcal, _ = self.sim.forward(layers)
                COST[i, j] = l2_misfit(dcal[0], self.dobs, std_noise=self.noise_std)
        elapsed_time = time.time() - start_time
        print(f"generated misfit map in {elapsed_time:.3f} seconds.")
        return -COST / self.sim.config.n_receivers, xgrid, ygrid
