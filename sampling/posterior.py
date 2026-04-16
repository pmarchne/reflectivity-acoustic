import numpy as np
from src.layers import update_layer_slice
from scipy.linalg import cho_factor, cho_solve


class FWIPosterior:
    """
    Evaluates the Bayesian Log-Posterior and its gradient for FWI.
    Ensures correct scaling between Prior and Likelihood.
    """

    def __init__(
        self, dobs, layers, sim, prior_mu, prior_cov, std_noise=1.0, beta=1.0, scale_factor=0.
    ):
        self.layers = layers
        self.sim = sim
        self.std_noise = float(std_noise)
        self.beta = float(beta)
        self.scale = scale_factor

        # Pre-process observed data
        self.dobs = np.asarray(dobs, dtype=float).squeeze()

        # Prior setup
        self.mu = np.asarray(prior_mu, dtype=float).ravel()
        self.cov = prior_cov
        # Using Cholesky for the inverse and log-det is more stable
        c, low = cho_factor(prior_cov)
        self._inv_cov = cho_solve((c, low), np.eye(self.mu.size))
        self._prior_logdet = 2 * np.sum(np.log(np.diag(c)))

    def _get_residual(self, layer, return_cache=False):
        """
        Computes synthetic data and the residual (d_syn - d_obs).
        """
        # Note: added return_cache=return_cache to the forward call
        dcal, cache = self.sim.forward(layer)
        dcal = dcal.squeeze()

        if self.scale > 0:
            dcal = dcal / self.scale

        residual = dcal - self.dobs
        return (residual, cache) if return_cache else residual

    def log_likelihood(self, model):
        """Calculates tempered log-likelihood: beta * ln p(d|m)"""
        lay = update_layer_slice(self.layers, vp_slice=model, start=1)
        residual = self._get_residual(lay)[0]
        n = residual.size
        # ln L = -0.5 * [ sum((res/sigma)^2) + n*ln(2*pi*sigma^2) ]
        ss = np.sum((residual / self.std_noise) ** 2)
        const = n * np.log(2.0 * np.pi * self.std_noise**2)

        ll = -0.5 * (ss + const)
        return self.beta * ll

    def grad_log_likelihood(self, model):
        """Calculates gradient log-likelihood via adjoint"""
        lay = update_layer_slice(self.layers, vp_slice=model, start=1)
        residual, cache = self._get_residual(lay, return_cache=True)

        # Compute adjoint gradient: G = [df/dm]^T * residual
        g_vp, _ = self.sim.gradient(residual=residual, layers=lay, cache=cache)

        # Apply chain rule
        scale = self.beta / (self.std_noise**2)
        if self.scale > 0:
            scale /= self.scale

        return -scale * g_vp[1:]

    def log_prior(self, model):
        """Calculates ln p(m) for a Gaussian prior."""
        diff = model - self.mu
        log_prior = -0.5 * (
            diff @ self._inv_cov @ diff + self._prior_logdet + self.mu.size * np.log(2.0 * np.pi)
        )
        # Add 'Soft Boundary' for gradient-based methods
        v_min, v_max = 1000.0, 7000.0
        for v in model:
            if v < v_min:
                log_prior -= 0.5 * (v - v_min)**2  # Sharp quadratic penalty
            elif v > v_max:
                log_prior -= 0.5 * (v - v_max)**2
        return log_prior

    def grad_log_prior(self, model):
        """Gradient of the Gaussian log-prior."""
        grad_vp = -self._inv_cov @ (model - self.mu)
        return grad_vp

    def log_posterior(self, model):
        """Combined log-target."""
        lp = self.log_prior(model)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(model)

    def grad_log_posterior(self, model):
        """Combined gradient"""
        return self.grad_log_prior(model) + self.grad_log_likelihood(model)

    def __call__(self, model):
        return self.log_posterior(model)
