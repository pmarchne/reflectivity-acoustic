import numpy as np
from scipy.special import expit, logit

class UnconstrainedTarget:
    """Adapter for wrapping FWIPosterior to unconstrained space."""

    def __init__(self, target):
        self.target = target
        self.v_min = target.v_min
        self.v_max = target.v_max
        self.range = target.range

    def z_to_m(self, z):
        return self.v_min + self.range * expit(z)

    def m_to_z(self, m):
        norm_m = np.clip((m - self.v_min) / self.range, 1e-15, 1.0 - 1e-15)
        return logit(norm_m)

    def log_jacobian_det(self, z):
        s = expit(z)
        # Sum along the last axis so it works for 1D vectors or 2D particle arrays
        return np.sum(
            np.log(self.range) + np.log(s + 1e-15) + np.log(1.0 - s + 1e-15),
            axis=-1,
        )

    def log_prior(self, z):
        m = self.z_to_m(z)
        return self.target.log_prior(m) + self.log_jacobian_det(z)

    def grad_log_prior(self, z):
        s = expit(z)
        m = self.z_to_m(z)
        grad_m = self.target.grad_log_prior(m)
        dm_dz = self.range * s * (1.0 - s)
        return grad_m * dm_dz + (1.0 - 2.0 * s)

    def log_likelihood(self, z):
        m = self.z_to_m(z)
        return self.target.log_likelihood(m)

    def grad_log_likelihood(self, z):
        s = expit(z)
        m = self.z_to_m(z)
        grad_m = self.target.grad_log_likelihood(m)
        dm_dz = self.range * s * (1.0 - s)
        return grad_m * dm_dz

    def log_posterior(self, z):
        return self.log_prior(z) + self.log_likelihood(z)

    def grad_log_posterior(self, z):
        return self.grad_log_prior(z) + self.grad_log_likelihood(z)

    def log_and_grad_post(self, z):
        """Single-pass evaluation of log-posterior and gradient in z-space."""
        s = expit(z)
        m = self.z_to_m(z)
        # 1. Single-pass physical likelihood + gradient (1 forward, 1 adjoint)
        ll, grad_ll_m = self.target.log_and_grad(m)
        lp_m = self.target.log_prior(m)
        grad_lp_m = self.target.grad_log_prior(m)
        dm_dz = self.range * s * (1.0 - s)
        log_det_J = np.sum(
            np.log(self.range) + np.log(s + 1e-15) + np.log(1.0 - s + 1e-15)
        )
        log_post = ll + lp_m + log_det_J
        grad_post = (grad_ll_m + grad_lp_m) * dm_dz + (1.0 - 2.0 * s)
        return log_post, grad_post


    def gmm_physical_to_unconstrained(self, means_m, sigma_m):
        ''' convert means and stds from gaussian mixture to unconstrained space (for sampling initialization)'''
        means_z = []
        Rs_z = []
        for m in means_m:
            z = self.m_to_z(m)
            means_z.append(z)
            norm_m = (m - self.v_min) / self.range
            dm_dz = self.range * norm_m * (1.0 - norm_m)
            std_z = sigma_m / dm_dz
            Rs_z.append(np.diag(std_z))
        return means_z, Rs_z


    def gmm_unconstrained_to_physical(self, means_z_traj, Rs_z_traj):
        ''' convert back means and stds from unconstrained space to physical velocity space'''
        means_m = [
            [self.z_to_m(comp_mean) for comp_mean in step_means] 
            for step_means in means_z_traj
        ]
        Rs_m = []
        for step_idx, step_means in enumerate(means_z_traj):
            step_Rs_m = []
            for comp_idx, z_comp in enumerate(step_means):
                s = expit(z_comp)
                dm_dz = self.range * s * (1.0 - s)
                R_z = Rs_z_traj[step_idx][comp_idx]
                cov_z = R_z @ R_z.T
                cov_m_comp = np.diag(dm_dz) @ cov_z @ np.diag(dm_dz)
                step_Rs_m.append(np.linalg.cholesky(cov_m_comp))
            Rs_m.append(step_Rs_m)
        return means_m, Rs_m