import numpy as np
from scipy.linalg import cho_factor, cho_solve
import logging

# Configure logger at module level
logger = logging.getLogger("FWIPosterior")
logger.setLevel(logging.INFO)

class FWIPosterior:
    """
    Evaluates the Bayesian Log-Posterior and its gradient for FWI.
    Ensures correct scaling between Prior and Likelihood.
    """
    def __init__(
        self, dobs, parametrization, sim, prior_mu, prior_cov, std_noise=1.0, beta=1.0, scale_factor=0.
    ):
        self.sim = sim
        self.std_noise = float(std_noise)
        self.beta = float(beta)
        self.scale = scale_factor
        self.param = parametrization
        # Pre-process observed data
        self.dobs = np.asarray(dobs, dtype=float).squeeze()
        # Prior setup
        self.mu = np.asarray(prior_mu, dtype=float).ravel()
        self.cov = prior_cov
        self.sigma = np.sqrt(np.diag(prior_cov))
        c, low = cho_factor(prior_cov)
        self._inv_cov = cho_solve((c, low), np.eye(self.mu.size))
        self._prior_logdet = 2 * np.sum(np.log(np.diag(c)))
        self.v_min, self.v_max = self.param.vp_bounds
        self.range = self.v_max - self.v_min

    def _get_residual(self, layer):
        """
        Computes synthetic data and the residual (d_syn - d_obs).
        """
        dcal = self.sim.forward(layer)
        dcal = dcal.squeeze()

        if self.scale > 0:
            dcal = dcal / self.scale

        residual = dcal - self.dobs
        return residual

    def l2_misfit(self, model):
        """Calculates tempered log-likelihood: beta * ln p(d|m)"""
        lay = self.param.build_layers(model)
        residual = self._get_residual(lay)
        ss = np.sum((residual / self.std_noise) ** 2)
        ll = 0.5 * ss
        return ll

    def log_likelihood(self, model):
        """Calculates tempered log-likelihood: beta * ln p(d|m)"""
        lay = self.param.build_layers(model)
        residual = self._get_residual(lay)
        n = residual.size
        # ln L = -0.5 * [ sum((res/sigma)^2) + n*ln(2*pi*sigma^2) ]
        ss = np.sum((residual / self.std_noise) ** 2)
        const = n * np.log(2.0 * np.pi * self.std_noise**2)
        ll = -0.5 * (ss + const)
        return self.beta * ll

    def log_prior(self, model):
        """Calculates ln p(m) for a Gaussian prior."""
        if np.any(model < self.v_min) or np.any(model > self.v_max):
            return -np.inf
        diff = model - self.mu
        gaussian_lp = -0.5 * (
            diff @ self._inv_cov @ diff + self._prior_logdet + self.mu.size * np.log(2.0 * np.pi)
        )
        return gaussian_lp

    def grad_log_prior(self, model):
        """Gradient of Gaussian log-prior + boundary penalty."""
        grad_gaussian = -self._inv_cov @ (model - self.mu)
        return grad_gaussian

    def grad_log_likelihood(self, model):
        """Calculates gradient log-likelihood via adjoint"""
        lay = self.param.build_layers(model)
        residual = self._get_residual(lay)
        grad = np.zeros(len(model))
        g_vp, _, g_h = self.sim.gradient(residual=residual, layers=lay)
        
        if self.param.invert_h == True:
            grad[0:self.param.n_vp] = g_vp[1:]
            grad[self.param.n_vp:] = g_h[1:-1]
        else :
            grad = g_vp[1:]
        # Apply chain rule
        scale = self.beta / (self.std_noise**2)
        if self.scale > 0:
            scale /= self.scale

        return -scale * grad
    
    def log_and_grad(self, model):
        """Calculates gradient log-likelihood via adjoint"""
        lay = self.param.build_layers(model)
        residual = self._get_residual(lay)
        n = residual.size
        ss = np.sum((residual / self.std_noise) ** 2)
        const = n * np.log(2.0 * np.pi * self.std_noise**2)
        log = -0.5 * self.beta * (ss + const)
    
        grad = np.zeros(len(model))
        g_vp, _, g_h = self.sim.gradient(residual=residual, layers=lay)
        
        if self.param.invert_h:
            grad[0:self.param.n_vp] = g_vp[1:]
            grad[self.param.n_vp:] = g_h[1:-1]
        else :
            grad = g_vp[1:]

        # Apply chain rule
        scale = self.beta / (self.std_noise**2)
        if self.scale > 0:
            scale /= self.scale
        grad *= -scale
        return log, grad

    def log_posterior(self, model):
        """Combined log-target."""
        return self.log_prior(model) + self.log_likelihood(model)

    def grad_log_posterior(self, model):
        """Combined gradient"""
        return self.grad_log_prior(model) + self.grad_log_likelihood(model)

    def log_and_grad_post(self, model):
        """Combined gradient"""
        ll, grad_ll = self.log_and_grad(model)
        lp = self.log_prior(model)
        grad_lp = self.grad_log_prior(model)
        return ll + lp, grad_ll + grad_lp

    def __call__(self, model):
        return self.log_posterior(model)
