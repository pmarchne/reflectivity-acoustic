import numpy as np
from scipy.stats import wasserstein_distance

def rbf_kernel_numpy(X, Y, gamma=1.0):
    """
    Computes the RBF (Gaussian) kernel matrix between X and Y using pure NumPy.
    X: shape (N, d), Y: shape (M, d)
    """
    X_sq = np.sum(X**2, axis=1).reshape(-1, 1)
    Y_sq = np.sum(Y**2, axis=1).reshape(1, -1)
    # Broadcasted squared Euclidean distance matrix
    sq_dists = X_sq + Y_sq - 2 * np.dot(X, Y.T)
    return np.exp(-gamma * sq_dists)

def compute_mmd_fast(X_scaled, Y_scaled, max_samples=5000):
    """
    Computes Maximum Mean Discrepancy (MMD) using NumPy vectorized kernel logic.
    Sub-samples down to `max_samples` to keep computation instant.
    """
    if X_scaled.shape[0] > max_samples:
        X_scaled = X_scaled[np.random.choice(X_scaled.shape[0], max_samples, replace=False)]
    if Y_scaled.shape[0] > max_samples:
        Y_scaled = Y_scaled[np.random.choice(Y_scaled.shape[0], max_samples, replace=False)]
        
    N, M = X_scaled.shape[0], Y_scaled.shape[0]
    
    mmd_total = 0.0
    scales = [0.5, 1.0, 2.0]
    
    for scale in scales:
        gamma = 1.0 / (2.0 * scale**2)
        XX = rbf_kernel_numpy(X_scaled, X_scaled, gamma=gamma)
        YY = rbf_kernel_numpy(Y_scaled, Y_scaled, gamma=gamma)
        XY = rbf_kernel_numpy(X_scaled, Y_scaled, gamma=gamma)
        
        # Unbiased MMD statistical estimator
        k_xx = (XX.sum() - np.trace(XX)) / (N * (N - 1))
        k_yy = (YY.sum() - np.trace(YY)) / (M * (M - 1))
        k_xy = XY.mean()
        
        mmd_total += (k_xx + k_yy - 2 * k_xy)
        
    return np.sqrt(np.maximum(mmd_total / len(scales), 0.0))

def compute_sliced_wasserstein_fast(X_scaled, Y_scaled, n_projections=200):
    """
    Optimized SWD using full array matrix multiplication and axis-sorting.
    """
    ndim = X_scaled.shape[1]
    
    # Equalize sample dimensions via fast uniform index alignment
    min_len = min(len(X_scaled), len(Y_scaled))
    idx_x = np.linspace(0, len(X_scaled) - 1, min_len, dtype=int)
    idx_y = np.linspace(0, len(Y_scaled) - 1, min_len, dtype=int)
    
    X_s = X_scaled[idx_x]
    Y_s = Y_scaled[idx_y]
    
    # Projections matrix setup
    projections = np.random.normal(size=(ndim, n_projections))
    projections /= np.linalg.norm(projections, axis=0)
    
    # Broadcast projections via a single dot product
    proj_X = X_s @ projections
    proj_Y = Y_s @ projections
    
    # Sort vectorized along the rows axis
    proj_X.sort(axis=0)
    proj_Y.sort(axis=0)
    
    return np.mean(np.abs(proj_X - proj_Y))

def run_diagnostics(results_ultranest, samples_approx, method_name="Approx Method"):
    ref_mean = np.array(results_ultranest['posterior']['mean'])
    ref_std = np.array(results_ultranest['posterior']['stdev'])
    samples_ref = results_ultranest['samples'] 
    ndim = len(ref_mean)
    
    print(f"\n=======================================================")
    print(f" DIAGNOSTICS: Reference vs {method_name}")
    print(f"=======================================================")
    print(f"{'Param':<6} | {'Ref Mean±Std':<20} | {'diff Mean':<9} | {'diff Std':<8} | {'1D W1':<8}")
    print("-" * 60)
    
    for i in range(ndim):
        m_ref, s_ref = ref_mean[i], ref_std[i]
        m_app, s_app = samples_approx[:, i].mean(), samples_approx[:, i].std()
        
        delta_mean = np.abs(m_ref - m_app)
        delta_std = np.abs(s_ref - s_app)
        w1_dist = wasserstein_distance(samples_ref[:, i], samples_approx[:, i])
        
        print(f"v_{i+1:<3} | {m_ref:>8.1f} ± {s_ref:<8.1f} | {delta_mean:<9.2f} | {delta_std:<8.2f} | {w1_dist:<8.2f}")
        
    print("-" * 60)
    
    # Pre-scale datasets once relative to your baseline distribution mapping
    mean_ref = samples_ref.mean(axis=0)
    std_ref = samples_ref.std(axis=0)
    X_scaled = (samples_ref - mean_ref) / std_ref
    Y_scaled = (samples_approx - mean_ref) / std_ref
    
    # Compute Metrics
    mmd_7d = compute_mmd_fast(X_scaled, Y_scaled)
    swd_7d = compute_sliced_wasserstein_fast(X_scaled, Y_scaled, n_projections=1000)
    
    print(f" >>> Joint 7D Maximum Mean Discrepancy (MMD):    {mmd_7d:.4f}")
    print(f" >>> Joint 7D Sliced Wasserstein (1000 projs):    {swd_7d:.4f}")
    print(f"=======================================================\n")