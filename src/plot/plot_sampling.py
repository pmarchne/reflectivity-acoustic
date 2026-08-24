import pickle
import numpy as np
from ultranest.plot import cornerplot, PredictionBand
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde, norm
from src.plot.plot_tools import set_plot_style
from sampling.fd_experiment import prepare_fd_model
from src.layers import create_layers_from_interfaces
from src.noise import add_noise_snr
from src.plot.diagnostics import run_diagnostics, get_swd

''' A bunch of functions for creating figures and generate statistics from the outputs of sampling methods '''

def setup_reference():
    with open('/home/marchnep/Results/beta1_fd/results_ultranest.pkl', 'rb') as file:
        results = pickle.load(file)
        print("results loaded !")
    samples = results['samples']
    print("shape samples", samples.shape)
    print(results['logz'])
    print(results['logzerr'])
    print(results['ncall'])
    vp_map = results['maximum_likelihood']['point']
    print("MAP vp:", vp_map)
    print(results['posterior'])
    return results


def corner_pdf(res, vp_true, save_fig=False):
    names = [f'$v_{i+1}$' for i in range(len(vp_true))]
    res['paramnames'] = names

    figure = cornerplot(res, plot_density=False, show_titles=True, smooth=0.5, hist_bin_factor=1.5,
        smooth1d=0.1, fill_contours=True, title_kwargs={"fontsize": 20}, levels=[0.3934, 0.6827, 0.9545, 0.9973],
        contour_kwargs={
            "linewidths": 1.5,
            "linestyles": ["-", "--", ":", "-."],
            "colors": ["#000000", "#000000", "#000000", "#000000"],
            "alpha": 0.9,
        },
        contourf_kwargs={
            "colors": ["#ffffff",  "#93cafd", "#4da6ff",  "#004c99", "#001f4d"],
            "alpha": 0.95,
        },
        hist_kwargs = {
            "color": "black",       # no fill
            "linewidth": 1.2,
        },
        figsize=(9,9),
    )
    ndim = len(names)
    axes = np.array(figure.axes).reshape((ndim, ndim))
    for i in range(ndim):
        ax = axes[i, i]
        for line in ax.lines:
            xdata = line.get_xdata()
            # Quantile lines are vertical
            if len(xdata) == 2 and xdata[0] == xdata[1]:
                line.set_color('purple')
                line.set_linewidth(1.2)
                line.set_linestyle("--")
            #ax.set_xlim(1000., 6000.)

        ax.axvline(vp_true[i], color="r", linestyle="-", alpha=0.7)
    # Loop over the histograms
    for yi in range(ndim):
        for xi in range(yi):
            ax = axes[yi, xi]
            ax.axvline(vp_true[xi], color="r", linestyle="-", lw=1.8, alpha=0.7)
            ax.axhline(vp_true[yi], color="r", linestyle="-", lw=1.8, alpha=0.7)
            ax.plot(vp_true[xi], vp_true[yi], "or", alpha=0.7, ms=10, zorder=4)
            #ax.set_xlim(1000., 6000.)
            #ax.set_ylim(1000., 6000.)

    for ax in figure.axes:
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=22,   # tick label size
            length=4,       # tick length
            width=1.0,
        )
        ax.tick_params(
            axis="both",
            which="minor",
            labelsize=20,
            length=2,
        )
        ax.xaxis.label.set_size(28)
        ax.yaxis.label.set_size(28)
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    legend_handles = [
        Line2D([0], [0], color="purple", lw=1.5, ls="--", label="hist. quantiles"),
        Line2D([0], [0], color="black", lw=1.5, ls="-.", label="39\\%"),
        Line2D([0], [0], color="black", lw=1.5, ls=":", label="68\\%"),
        Line2D([0], [0], color="black", lw=1.5, ls="--", label="95\\%"),
        Line2D([0], [0], color="black", lw=1.5, ls="-", label="99\\%"),
        Line2D([0], [0], color="r", lw=1.5, ls="-", label="Reference"),
    ]

    figure.legend(
        handles=legend_handles,
        loc="center right",
        frameon=False,
        fontsize=28,
    )
    path = ''
    if save_fig:
        plt.savefig(path + 'corner_plot.pdf')
        plt.close()
    else:
        plt.show()


def marginals1D_mog(samples_ref, means, Rs, weights, it_num=-1, save_fig=False, seed=0, color='red', lbl='tmp'):
    ndim = samples_ref.shape[1]
    fig, axes = plt.subplots(1, ndim, figsize=(4 * ndim + 6, 4))
    nkde = 1000
    vmin, vmax = 1000., 6000.
    x = np.linspace(vmin, vmax, nkde)
    
    # Extract parameters for the current iteration
    it_means = means[it_num]       # List of 4 arrays of shape (7,)
    it_Rs = Rs[it_num]             # List of 4 arrays of shape (7, 7)
    it_weights = weights[it_num]   # List/array of 4 weights
    n_components = len(it_weights)

    for i in range(ndim):
        counts, bins = np.histogram(samples_ref[:, i], bins=100, density=True)
        max_count = counts.max()
        # Plot Reference
        nbins = 60
        axes[i].hist(
            samples_ref[:, i],
            bins=nbins,
            density=True,
            histtype="stepfilled",
            edgecolor="black",
            facecolor=(0, 0, 0, 0.2),  # RGBA black with transparency
            linewidth=1.2,
            label="Reference",
        )
        axes[i].set_title(r"$v_{" + str(i+1) + "}$", fontsize=36)
        axes[i].set_ylim(0, 1.5*max_count)
        # Compute Analytical Mixture of Gaussians 1D PDF
        mog_pdf = np.zeros_like(x)
        for k in range(n_components):
            mu_k = it_means[k][i]
            # Compute variance from Cholesky factor: Sigma = R @ R.T
            cov_matrix = it_Rs[k] @ it_Rs[k].T
            sigma_k = np.sqrt(cov_matrix[i, i])
            mog_pdf += it_weights[k] * norm.pdf(x, mu_k, sigma_k)
            
        axes[i].plot(x, mog_pdf, color=color, linestyle='--', lw=3, label=lbl)
        axes[i].set_yticks([])          
        axes[i].set_yticklabels([]) 
        axes[i].tick_params(axis='x', labelsize=26)
        axes[i].set_xlim(vmin, vmax)

    axes[2].legend(fontsize=24, frameon=False)
    axes[0].set_xlim(2200, 3200)
    axes[1].set_xlim(2000, 4000)
    axes[2].set_xlim(1500, 3500)
    plt.tight_layout()
    path = '/home/marchnep/Documents/Gitlab_repos/2026_MARCHNER_UQFWI/Fig/fwi_7vp/'
    if save_fig:
        plt.savefig(path + f'mixGVI_marginals_it{it_num}_N{n_components}_seed{seed}.pdf')
        plt.close()
    else:
        plt.show()


# plot 1D marginals of particle method e.g. svgd
def marginals1D_particles(samples_ref, samples_part, it_num=-1, save_fig=False, seed=0):
    ndim = samples_ref.shape[1]
    s_part = samples_part[it_num]
    fig, axes = plt.subplots(1, ndim, figsize=(4*ndim+2, 4))
    nkde = 500
    vmin, vmax = 1000., 6000.
    for i in range(ndim):
        # kde = gaussian_kde(samples_ref[:, i])
        x = np.linspace(vmin, vmax, nkde)
        kde_part = gaussian_kde(s_part[:, i])
        nbins = 60
        counts, bins = np.histogram(samples_ref[:, i], bins=nbins, density=True)
        max_count = counts.max()
        axes[i].hist(
            samples_ref[:, i],
            bins=nbins,
            density=True,
            histtype="stepfilled",
            edgecolor="black",
            facecolor=(0, 0, 0, 0.2),  # RGBA black with transparency
            linewidth=1.2,
            label="Reference",
        )
        axes[i].set_ylim(0, 1.5*max_count)
        #axes[i].hist(samples_ref[:, i], bins=nbins, density=True, alpha=0.7, label='Reference', color="black", edgecolor="black", linewidth=1.2)
        axes[i].set_title(r"$v_{" + str(i+1) + "}$", fontsize=36)
        #axes[i].set_ylim(0, 1.5 * nbins)
        axes[i].plot(x, kde_part(x), lw=3., label='SVGD', color='green', linestyle='--')
        axes[i].set_yticks([]) 
        axes[i].set_yticklabels([]) 
        axes[i].tick_params(axis='x', labelsize=26)
        axes[i].set_xlim(vmin, vmax)
    axes[2].legend(fontsize=24, frameon=False)
    axes[0].set_xlim(2200, 3200)
    axes[1].set_xlim(2000, 4000)
    axes[2].set_xlim(1500, 3500)
    plt.tight_layout()
    path = '/home/marchnep/Documents/Gitlab_repos/2026_MARCHNER_UQFWI/Fig/fwi_7vp/'
    if save_fig:
        plt.savefig(path + f'svgd_marginals_it{it_num}_seed{seed}.pdf')
        plt.close()
    else:
        plt.show()



def plot_band(res, save_fig):
    path = "FD_comparison/data/seis_v3_nofs"
    bayes, param = prepare_fd_model(file_path=path, seed=42, debug=False)
    time = bayes.sim.param.time

    z_int = np.array([0.0, 100.0, 200.0, 275.0, 375.0, 400.0, 500.0, 550.0, 700.0])
    vps_ref = np.array([1505.0, 2700.0, 3200.0, 1900.0, 4200.0, 3800.0, 2200.0, 4500.0])
    rhos = np.full_like(vps_ref, 2000.0)

    lays_ref = create_layers_from_interfaces(z_int, vps_ref, rhos)
    d_obs = bayes.sim.forward(lays_ref)
    d_obs_final, std_noise = add_noise_snr(d_obs, snr_db=10, seed=42)
    print("dobs shape", d_obs.shape)
    print("shape plot",d_obs[0, 0, :].shape)
    print("time shape", time.shape)

    band0 = PredictionBand(time)
    band1 = PredictionBand(time)
    band2 = PredictionBand(time)
    band3 = PredictionBand(time)
    samples = res['samples']
    ns = samples.shape[0]
    print('number of samples:', ns)
    # go through the solutions

    traces = [1, 3, 5, 7]
    idx = np.random.choice(ns, 30000, replace=False)
    for v in samples[idx]:
        # compute for each time the y value
        VP = np.array([1505., v[0], v[1], v[2], v[3], v[4], v[5], v[6]])
        lays = create_layers_from_interfaces(z_int, VP, rhos)
        dcal = bayes.sim.forward(lays)
        # Add observational noise
        # noise0 = np.random.normal(0, std_noise, size=dcal[0, traces[0], :].shape)
        #band0.add(dcal[0, traces[0], :] + noise0)
        band0.add(dcal[0, traces[0], :])
        band1.add(dcal[0, traces[1], :])
        band2.add(dcal[0, traces[2], :])
        band3.add(dcal[0, traces[3], :])

    fig, axs = plt.subplots(
        4, 1,
        figsize=(5, 8),
        sharex=True
    )

    bands = [band0, band1, band2, band3]

    for ax, band, tr in zip(axs, bands, traces):
        plt.sca(ax) 
        ax.plot(time, d_obs_final[0, tr, :], marker='.', color='black', ls=' ',alpha=0.35, markersize=5, label=r'$d_{\mathrm{obs}}$')
        band.line(color='b', label='Median')
        band.shade(color='b', alpha=0.5)           # default credible interval
        band.shade(q=0.4985, color='b', alpha=0.25)  # 99.7% credible interval
        ax.set_ylim([-1.5, 1.5])
        ax.set_xlim([0., 1.024])
        ax.grid(alpha=0.3)

    axs[0].legend(loc='upper right', fontsize=14, bbox_to_anchor=(1.02, 1.0))
    axs[-1].set_xlabel("Time [s]")

    plt.tight_layout()
    path_save = '' #'/Documents/Fig/fwi_7vp/'
    if save_fig:
        plt.savefig(path_save + 'data_UQ.pdf')
        plt.close()
    else:
        plt.show()


def get_MoG(means, Rs, weights, target_it, dim):
    it_weights = weights[target_it]
    K = len(it_weights)
    comp_idx = np.random.choice(K, size=100000, p=it_weights)
                    
    mog_samples = np.zeros((100000, dim))
    for k in range(K):
        m = (comp_idx == k)
        if np.sum(m) > 0:
            mog_samples[m] = means[target_it][k] + np.random.normal(size=(np.sum(m), dim)) @ Rs[target_it][k].T
    return mog_samples


def marginals1D_gvi(samples_ref, means, Rs, means2, Rs2, map, hess_inv, it_num=-1, save_fig=False):
    ndim = samples_ref.shape[1]
    fig, axes = plt.subplots(1, ndim, figsize=(4 * ndim + 6, 4))
    nkde = 1000
    vmin, vmax = 1000., 6000.
    x = np.linspace(vmin, vmax, nkde)
    
    # Extract parameters for the current iteration
    mean1 = means[it_num][0]
    R1 = Rs[it_num][0]
    mean2 = means2[it_num][0]
    R2 = Rs2[it_num][0]

    for i in range(ndim):
        counts, bins = np.histogram(samples_ref[:, i], bins=100, density=True)
        max_count = counts.max()
        lbl_laplace = 'Laplace' if i == 0 else ""
        lbl_gvi1 = '$\mathbf{m}_0 = 2500$' if i == 1 else ""
        lbl_gvi2 = '$\mathbf{m}_0 = 2800$' if i == 2 else ""
        lbl_ref = 'Reference' if i == 5 else ""
        # Plot Reference
        nbins = 60
        axes[i].hist(
            samples_ref[:, i],
            bins=nbins,
            density=True,
            histtype="stepfilled",
            edgecolor="black",
            facecolor=(0, 0, 0, 0.2),  # RGBA black with transparency
            linewidth=1.2,
            label=lbl_ref
        )
        axes[i].set_title(r"$v_{" + str(i+1) + "}$", fontsize=36)
        axes[i].set_ylim(0, 1.5*max_count)
        # Compute Analytical Mixture of Gaussians 1D PDF
        mu = mean1[i]
        cov = R1 @ R1.T
        sigma = np.sqrt(cov[i, i])
        gvi1 = norm.pdf(x, mu, sigma)

        mu2 = mean2[i]
        cov2 = R2 @ R2.T
        sigma2 = np.sqrt(cov2[i, i])
        gvi2 = norm.pdf(x, mu2, sigma2)

        mu3 = map[i]
        sigma3 = np.sqrt(hess_inv[i, i])
        gvi3 = norm.pdf(x, mu3, sigma3)
        

        axes[i].plot(x, gvi3, 'k-.', lw=3, label=lbl_laplace)
        axes[i].plot(x, gvi1, color='blue', linestyle=':', lw=3, label=lbl_gvi1)
        axes[i].plot(x, gvi2, color='teal', linestyle='--', lw=3, label=lbl_gvi2)
        axes[i].legend(fontsize=24, frameon=False)
        axes[i].set_yticks([])          
        axes[i].set_yticklabels([]) 
        axes[i].tick_params(axis='x', labelsize=26)
        axes[i].set_xlim(vmin, vmax)

    axes[0].set_xlim(2200, 3200)
    axes[1].set_xlim(2000, 4000)
    axes[2].set_xlim(1500, 3500)
    plt.tight_layout()
    path = '/home/marchnep/Documents/Gitlab_repos/2026_MARCHNER_UQFWI/Fig/fwi_7vp/'
    if save_fig:
        plt.savefig(path + f'GVI_marginals_it{it_num}.pdf')
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    vp_true = np.array([2700.0, 3200.0, 1900.0, 4200.0, 3800.0, 2200.0, 4500.0])
    dim = len(vp_true)
    set_plot_style()
    res = setup_reference()
    corner_pdf(res, vp_true, save_fig=False)
    
    #rng = np.random.default_rng(42)
    obs_path = "FD_comparison/data/seis_v3_nofs"
    bayes, param = prepare_fd_model(file_path=obs_path, seed=42, debug=False)

    logZ = 141.86576429722984
    ns = 2000
    target_it = 200
    target_it_gvi = 200
    plot_svgd = False
    plot_mixgvi = False
    plot_gvi = False
    plot_reference = True
    save_fig = False
    folder = 'results_mog_svgd/'
    seed = 0

    if plot_reference:
        corner_pdf(res, vp_true, save_fig=save_fig)
        # plot_band(res, save_fig=save_fig)    

    if plot_gvi:
        with open('res_gvi_std500_mu2500.pkl', 'rb') as fp:
            res_gvi1 = pickle.load(fp)
            mean1, Rs1, w1 = res_gvi1[0], res_gvi1[1], res_gvi1[2]
        with open('res_gvi_std500_mu2800.pkl', 'rb') as fp:
            res_gvi2 = pickle.load(fp)
            mean2, Rs2, w2 = res_gvi2[0], res_gvi2[1], res_gvi2[2]

        map = np.array([2700.692, 3062.192, 2014.517, 3081.276, 3697.679, 2174.89,  3283.172])
        hess_inv_laplace = np.array([
            [ 1846.731, -2671.837,    43.451, -1651.138,  -682.262,     5.521,    26.808],
            [-2671.837, 19537.423, -2770.513, -4958.137,  2033.839, -2074.682,   511.138],
            [   43.451, -2770.513,  9254.517, -50516.805, -5234.732, -2949.832,   524.670],
            [-1651.138, -4958.137, -50516.805, 387034.687,  8690.691, 12263.718, -2221.569],
            [ -682.262,  2033.839, -5234.732,  8690.691, 33611.289, -3813.489,    86.308],
            [    5.521, -2074.682, -2949.832, 12263.718, -3813.489, 29890.679, -5341.640],
            [   26.808,   511.138,   524.670, -2221.569,    86.308, -5341.640, 38241.855]
        ])

        # 1. Standard Deviations (square roots of the diagonal elements)
        std_devs = np.sqrt(np.diag(hess_inv_laplace))
        print("Standard Deviations (std):", std_devs)
        marginals1D_gvi(res['samples'], mean1, Rs1, mean2, Rs2, map, hess_inv_laplace, it_num=target_it_gvi, save_fig=save_fig)
        gvi1_samples = get_MoG(mean1, Rs1, w1, target_it_gvi, dim)
        _, _ = run_diagnostics(res, gvi1_samples, method_name=f"GVI 1 (it={target_it_gvi})", verbose=True)

        gvi2_samples = get_MoG(mean2, Rs2, w2, target_it_gvi, dim)
        _, _ = run_diagnostics(res, gvi2_samples, method_name=f"GVI 2 (it={target_it_gvi})", verbose=True)
        samples_laplace = np.random.multivariate_normal(mean=map, cov=hess_inv_laplace, size=50000)
        _, _ = run_diagnostics(res, samples_laplace, method_name=f"Laplace approx.", verbose=True)

    if plot_svgd:
        with open('res_svgd_unconst_rng'+str(seed)+'.pkl', 'rb') as fp:
            res_load = pickle.load(fp)
            res_svgd, kl_hist = res_load[0], res_load[1]
            print("results loaded !")
        _, _ = run_diagnostics(res, res_svgd[target_it], method_name=f"svgd (it={target_it})", verbose=True)
        marginals1D_particles(res['samples'], res_svgd, it_num=target_it, save_fig=save_fig, seed=seed)
    
    if plot_mixgvi:
        with open('test_unconst_MoG5_rng'+str(seed)+'.pkl', 'rb') as fp:
            res_mixgvi5 = pickle.load(fp)
            means5, Rs5, weights5, kl5, dt_w5 = res_mixgvi5[0], res_mixgvi5[1], res_mixgvi5[2], res_mixgvi5[3], res_mixgvi5[4]
            print("results 5 mix gvi loaded !")
        MoG5_samples = get_MoG(means5, Rs5, weights5, target_it, dim)
        _, _ = run_diagnostics(res, MoG5_samples, method_name=f"MoG VI (it={target_it_gvi})", verbose=True)
        marginals1D_mog(res['samples'], means5, Rs5, weights5, it_num=target_it_gvi, save_fig=save_fig, seed=seed, color='orange', lbl='$K=5$')

        with open('test_unconst_MoG10_rng'+str(seed)+'.pkl', 'rb') as fp:
            res_mixgvi = pickle.load(fp)
            means, Rs, weights, kl, dt_w = res_mixgvi[0], res_mixgvi[1], res_mixgvi[2], res_mixgvi[3], res_mixgvi[4]
            print("results 10 mix gvi loaded !")

        MoG10_samples = get_MoG(means, Rs, weights, target_it_gvi, dim)  
        _, _ = run_diagnostics(res, MoG10_samples, method_name=f"MoG VI (it={target_it_gvi})", verbose=True)
        marginals1D_mog(res['samples'], means, Rs, weights, it_num=target_it_gvi, save_fig=save_fig, seed=seed, color='red', lbl='$K=10$')


    SWD5, SWD10, SWD_S = [], [], []
    n_proj = 500
    # Loop over iterations to track metrics over time
    for it in range(target_it):
        # 1. Generate samples for the current iteration 'it'
        mog5_samples_it = get_MoG(means5, Rs5, weights5, it, dim)    
        mog10_samples_it = get_MoG(means, Rs, weights, it, dim)
        # 2. Run diagnostics and unpack both metrics
        swd_k5 = get_swd(res, mog5_samples_it, n_proj = n_proj)
        swd_k10 = get_swd(res, mog10_samples_it, n_proj = n_proj)
        swd_s = get_swd(res, res_svgd[it], n_proj = n_proj)
        print(it)
        SWD5.append(swd_k5)
        SWD10.append(swd_k10)
        SWD_S.append(swd_s)

    # 4. Plotting the convergence metrics over iterations
    iterations = range(target_it)
    plt.figure(figsize=(8, 5))
    plt.plot(iterations, SWD5, label='$K=5$')
    plt.plot(iterations, SWD10, label='$K=10$')
    plt.plot(iterations, SWD_S, label='SVGD')
    plt.xlabel('Iteration')
    plt.ylabel('Distance')
    plt.title('SWD')
    plt.legend()
    plt.show()

    #np.savez(
    #    f"swd_convergence_seed{seed}_fine.npz",
    #    SWD5=np.array(SWD5),
    #    SWD10=np.array(SWD10),
    #    SWD_S=np.array(SWD_S)
    #)