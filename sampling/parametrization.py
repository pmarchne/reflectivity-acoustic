import numpy as np
from scipy.stats import truncnorm
from src.layers import to_arrays, update_layer_slice
from src.utilities import timer

class ModelParameterization:
    def __init__(self, layers, invert_vp=True, invert_h=False, startv=1, starth=0, prior_mode='uniform', mu=None, cov=None):
        self.layers = layers
        self.invert_vp = invert_vp
        self.invert_h = invert_h
        self.startv = startv
        self.starth = starth
        self.prior_mode = prior_mode
        # Bounds
        self.vp_bounds = (1000.0, 6000.0)
        self.h_bounds = (10.0, 2000.0)

        if (prior_mode == 'gaussian') and (mu is None or cov is None):
            raise ValueError('mean and covariance not specified !')

        hs, vps, _ = to_arrays(layers)
        self.n_vp = len(vps[startv:]) if invert_vp else 0
        self.n_h = ( len(hs[starth:]) - 1 ) if invert_h else 0

        # Gaussian Statistics
        self.mu = mu if mu is not None else np.zeros(self.ndim)
        self.sigma = np.sqrt(np.diag(cov)) if cov is not None else np.ones(self.ndim)

    @property
    def ndim(self):
        return self.n_vp + self.n_h

    def split(self, model):
        """
        Split model vector into parameter blocks.
        """
        i = 0
        vp = None
        h = None
        if self.invert_vp:
            vp = model[i:i+self.n_vp]
            i += self.n_vp
        if self.invert_h:
            h = model[i:i+self.n_h]
            i += self.n_h
        return vp, h

    def build_layers(self, model):
        """
        Convert model vector -> updated layers.
        """
        vp, h = self.split(model)

        return update_layer_slice(
            self.layers,
            vp_slice=vp,
            hs_slice=h,
            startv=self.startv,
            starth=self.starth
        )
    

    def prior_transform(self, cube):
        cube = np.asarray(cube)
        params = np.empty_like(cube)
        i = 0

        # Helper to map a slice based on mode
        def map_slice(start, n, bounds, mu, sigma):
            slc = slice(start, start + n)
            if self.prior_mode == 'uniform':
                return cube[slc] * (bounds[1] - bounds[0]) + bounds[0]
            else:
                # Truncated Gaussian mapping
                a = (bounds[0] - mu[slc]) / sigma[slc]
                b = (bounds[1] - mu[slc]) / sigma[slc]
                return truncnorm.ppf(cube[slc], a, b, loc=mu[slc], scale=sigma[slc])

        if self.invert_vp:
            params[i:i+self.n_vp] = map_slice(i, self.n_vp, self.vp_bounds, self.mu, self.sigma)
            i += self.n_vp

        if self.invert_h:
            params[i:i+self.n_h] = map_slice(i, self.n_h, self.h_bounds, self.mu, self.sigma)
            i += self.n_h

        return params
    
    def draw_from_prior(self):
        """
        Draws a random model from the prior distribution.
        """
        unit_cube = np.random.rand(self.ndim)
        return self.prior_transform(unit_cube)


def diagnostic_checks(bayes, prior_transform, m_ref=None):
    """
    Performs sanity checks on the Likelihood, Prior, and Parameter Mapping.
    """
    mu = bayes.mu
    sigma = np.sqrt(np.diag(bayes.cov))
    print("\n" + "=" * 50)
    print(f"{'BAYESIAN FWI DIAGNOSTIC CHECKS':^50}")
    print("=" * 50)

    # 1. Check at Prior Mean
    with timer("prior eval"):
        lp_mu = bayes.log_prior(mu)
    with timer("likelihood eval"):
        ll_mu = bayes.log_likelihood(mu)
    print(f"[*] Prior Mean (mu): {mu}")
    print(f"    > log_prior:      {lp_mu:.4f}")
    print(f"    > log_likelihood: {ll_mu:.4f}")

    # 2. Check at mu + sigma (Sensitivity Check)
    vp_test = mu + sigma
    with timer("likelihood eval"):
        ll_sigma = bayes.log_likelihood(vp_test)
    delta_ll = ll_sigma - ll_mu
    print(f"[*] Shifted (mu + sigma): {vp_test}")
    print(f"    > log_likelihood: {ll_sigma:.4f}")
    print(f"    > Delta LL:       {delta_ll:.4f}")

    # 3. Reference Check (The 'Target' Score)
    if m_ref is not None:
        with timer("likelihood eval"):
            ll_ref = bayes.log_likelihood(m_ref)
        print(f"[*] True Reference:      {m_ref}")
        print(f"    > log_likelihood: {ll_ref:.4f}")
        print(f"    > Distance to Target: {ll_ref - ll_mu:.4f} (log-units)")

    # 4. Range & Sensitivity Validation
    if np.isclose(ll_mu, ll_sigma, rtol=1e-5):
        print("\n!! WARNING: Likelihood is FLAT. The data is not sensitive to Vp.")
        print("   Check if noise_std is too large or if the simulation is running.")

    if np.abs(delta_ll) > 1e6:
        print("\n!! WARNING: Likelihood is extremely SHARP.")
        print(
            "   This might lead to poor sampling. Consider increasing noise_std or beta."
        )

    # 5. Prior Transform Mapping (Unit Cube -> Physical)
    test_cube = np.array([0.5] * len(mu))
    transformed = prior_transform(test_cube)
    print("\n[*] Prior Mapping Check:")
    print(f"    > Unit Cube [0.5] maps to: {transformed}")

    # Check if mapping stays in bounds
    if np.any(transformed < 50) or np.any(transformed > 7000):
        print("    !! ERROR: Prior transform is mapping outside [1000, 7000]!")
    else:
        print("    > Range check: PASS")

    print("=" * 50 + "\n")




    '''def prior_transform(self, cube):

        cube = np.asarray(cube)
        params = np.empty_like(cube)

        i = 0

        if self.invert_vp:

            params[i:i+self.n_vp] = (
                cube[i:i+self.n_vp]
                * (self.vp_max - self.vp_min)
                + self.vp_min
            )

            i += self.n_vp

        if self.invert_h:

            params[i:i+self.n_h] = (
                cube[i:i+self.n_h]
                * (self.h_max - self.h_min)
                + self.h_min
            )

            i += self.n_h

        return params'''