import pickle
import numpy as np
from ultranest.plot import cornerplot, PredictionBand
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm
from src.plot.plot_tools import set_plot_style
from sampling.fd_experiment import prepare_fd_model
from src.layers import create_layers_from_interfaces
from src.noise import add_noise_snr
from src.plot.diagnostics import run_diagnostics

def setup_reference():
    with open('/home/marchnep/Results/beta1/ultranest_results_beta1_fd_vp_N2000/results_ultranest.pkl', 'rb') as file:
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


def corner_pdf(res, vp_true):
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
    path = '/home/marchnep/Documents/Gitlab_repos/2026_MARCHNER_UQFWI/Fig/fwi_7vp/'
    plt.savefig(path + 'corner_plot_7vp.pdf')
    plt.close()
    #plt.show()

'''fig, axes = plt.subplots(7, 1, figsize=(10, 15), sharex=True)
for i in range(7):
    axes[i].plot(samples[:, i], alpha=0.5)
    axes[i].set_ylabel(f"Vp_{i+1}")
plt.xlabel("Iteration")
plt.tight_layout()
plt.show()'''

def marginals1D_mog(samples_ref, means, Rs, weights, it_num=-1):
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
        axes[i].hist(samples_ref[:, i], bins=80, density=True, alpha=0.4, label='Reference', color="blue")
        axes[i].set_title(r"$v_{" + str(i+1) + "}$", fontsize=24)
        axes[i].set_ylim(0, 1.2 * max_count)
        # Compute Analytical Mixture of Gaussians 1D PDF
        mog_pdf = np.zeros_like(x)
        for k in range(n_components):
            mu_k = it_means[k][i]
            # Compute variance from Cholesky factor: Sigma = R @ R.T
            cov_matrix = it_Rs[k] @ it_Rs[k].T
            sigma_k = np.sqrt(cov_matrix[i, i])
            mog_pdf += it_weights[k] * norm.pdf(x, mu_k, sigma_k)
            
        axes[i].plot(x, mog_pdf, 'r-', lw=3, label='MoG VI')
        axes[i].set_yticks([])          
        axes[i].set_yticklabels([]) 
        axes[i].tick_params(axis='x', labelsize=20)
        axes[i].set_xlim(vmin, vmax)

    axes[0].legend(fontsize=20)
    plt.tight_layout()
    plt.show()


def corner_plot_mog(means, Rs, weights, vp_true, it_num=-1, n_samples=50000):
    # --- 1. Sample from MoG at the given iteration ---
    it_means = means[it_num]
    it_Rs = Rs[it_num]
    it_weights = weights[it_num]
    n_components = len(it_weights)
    
    comp_indices = np.random.choice(n_components, size=n_samples, p=it_weights)
    ndim = len(vp_true)
    mog_samples = np.zeros((n_samples, ndim))
    
    for k in range(n_components):
        mask = (comp_indices == k)
        n_k = np.sum(mask)
        if n_k > 0:
            z = np.random.normal(size=(n_k, ndim))
            mog_samples[mask] = it_means[k] + z @ it_Rs[k].T

    # --- 2. Mock an Ultranest results dictionary correctly ---
    names = [f'$v_{i+1}$' for i in range(ndim)]
    uniform_weights = np.ones(n_samples) / n_samples
    
    mock_res = {
        'paramnames': names,
        'weighted_samples': {
            'points': mog_samples,
            'weights': uniform_weights
        }
    }

    # --- 3. Generate Plot Using Your Exact Style Specifications ---
    figure = cornerplot(mock_res, plot_density=False, show_titles=True, smooth=0.5, hist_bin_factor=1.5, smooth1d=0.1, fill_contours=True, title_kwargs={"fontsize": 20}, levels=[0.3934, 0.6827, 0.9545, 0.9973],
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
            "color": "black",
            "linewidth": 1.2,
        },
        figsize=(9,9),
    )
    
    axes = np.array(figure.axes).reshape((ndim, ndim))
    for i in range(ndim):
        ax = axes[i, i]
        for line in ax.lines:
            xdata = line.get_xdata()
            if len(xdata) == 2 and xdata[0] == xdata[1]:
                line.set_color('purple')
                line.set_linewidth(1.2)
                line.set_linestyle("--")

        ax.axvline(vp_true[i], color="r", linestyle="-", alpha=0.7)
        
    for yi in range(ndim):
        for xi in range(yi):
            ax = axes[yi, xi]
            ax.axvline(vp_true[xi], color="r", linestyle="-", lw=1.8, alpha=0.7)
            ax.axhline(vp_true[yi], color="r", linestyle="-", lw=1.8, alpha=0.7)
            ax.plot(vp_true[xi], vp_true[yi], "or", alpha=0.7, ms=10, zorder=4)

    for ax in figure.axes:
        ax.tick_params(axis="both", which="major", labelsize=22, length=4, width=1.0)
        ax.tick_params(axis="both", which="minor", labelsize=20, length=2)
        ax.xaxis.label.set_size(28)
        ax.yaxis.label.set_size(28)
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color="purple", lw=1.5, ls="--", label="hist. quantiles"),
        Line2D([0], [0], color="black", lw=1.5, ls="-.", label="39\\%"),
        Line2D([0], [0], color="black", lw=1.5, ls=":", label="68\\%"),
        Line2D([0], [0], color="black", lw=1.5, ls="--", label="95\\%"),
        Line2D([0], [0], color="black", lw=1.5, ls="-", label="99\\%"),
        Line2D([0], [0], color="r", lw=1.5, ls="-", label="Reference"),
    ]

    figure.legend(handles=legend_handles, loc="center right", frameon=False, fontsize=28)
    
    #path = '/home/marchnep/Documents/Gitlab_repos/2026_MARCHNER_UQFWI/Fig/fwi_7vp/'
    #plt.savefig(path + f'corner_plot_mog_it{it_num}.pdf')
    #plt.close()
    plt.show()

# plot 1D marginals of particle method e.g. svgd
def marginals1D_particles(samples_ref, samples_part, it_num=-1):
    ndim = samples_ref.shape[1]
    s_part = samples_part[it_num]
    fig, axes = plt.subplots(1, ndim, figsize=(4*ndim+6, 4))
    nkde = 500
    vmin, vmax = 1000., 6000.
    for i in range(ndim):
        # kde = gaussian_kde(samples_ref[:, i])
        x = np.linspace(vmin, vmax, nkde)
        kde_part = gaussian_kde(s_part[:, i])
        counts, bins = np.histogram(samples_ref[:, i], bins=100, density=True)
        max_count = counts.max()
        axes[i].hist(samples_ref[:, i], bins=80, density=True, alpha=0.4, label='Reference', color="blue")
        axes[i].set_title(r"$v_{" + str(i+1) + "}$", fontsize=24)
        axes[i].set_ylim(0, 1.5 * max_count)
        axes[i].plot(x, kde_part(x), lw=2, label='SVGD', color='green', linestyle='--')
        axes[i].set_title(r"$v_{" + str(i+2) + "}$", fontsize=24)
        axes[i].set_yticks([]) 
        axes[i].set_yticklabels([]) 
        axes[i].tick_params(axis='x', labelsize=20)
        axes[i].set_xlim(vmin, vmax)
    axes[0].legend(fontsize=20)
    #axes[0].set_xlim(1700, 2300)
    plt.tight_layout()
    plt.show()


def plot_band(res):
    path = "FD_comparison/data/seis_v3_nofs"
    bayes, param = prepare_fd_model(file_path=path, seed=42, debug=False)
    time = bayes.sim.param.time

    z_int = np.array([0.0, 100.0, 200.0, 275.0, 375.0, 400.0, 500.0, 550.0, 700.0])
    vps_ref = np.array([1505.0, 2700.0, 3200.0, 1900.0, 4200.0, 3800.0, 2200.0, 4500.0])
    rhos = np.full_like(vps_ref, 2000.0)

    band0 = PredictionBand(time)
    band1 = PredictionBand(time)
    band2 = PredictionBand(time)
    band3 = PredictionBand(time)
    samples = res['samples']
    ns = samples.shape[0]
    print('number of samples:', ns)
    # go through the solutions

    traces = [1, 3, 5, 7]
    idx = np.random.choice(ns, 3000, replace=False)
    for v in samples[idx]:
        # compute for each time the y value
        VP = np.array([1505., v[0], v[1], v[2], v[3], v[4], v[5], v[6]])
        lays = create_layers_from_interfaces(z_int, VP, rhos)
        dcal = bayes.sim.forward(lays)
        band0.add(dcal[0, traces[0], :])
        band1.add(dcal[0, traces[1], :])
        band2.add(dcal[0, traces[2], :])
        band3.add(dcal[0, traces[3], :])

    lays_ref = create_layers_from_interfaces(z_int, vps_ref, rhos)
    d_obs = bayes.sim.forward(lays_ref)
    d_obs_final, std_noise = add_noise_snr(d_obs, snr_db=10, seed=42)
    print("dobs shape", d_obs.shape)
    print("shape plot",d_obs[0, 0, :].shape)
    print("time shape", time.shape)
    fig, axs = plt.subplots(
        4, 1,
        figsize=(5, 8),
        sharex=True
    )

    bands = [band0, band1, band2, band3]

    for ax, band, tr in zip(axs, bands, traces):
        plt.sca(ax) 
        band.line(color='b', label='Mean')
        band.shade(color='b', alpha=0.5)           # default credible interval
        band.shade(q=0.5, color='blue', alpha=0.25)  # 99% credible interval
        #ax.plot(time, d_obs[0, tr, :], 'k:', label='Reference')
        ax.plot(time, d_obs_final[0, tr, :], 'k:', alpha=0.7, label='Observations')
        ax.set_ylim([-1.5, 1.5])
        ax.set_xlim([0., 1.024])
        ax.grid(alpha=0.3)

    axs[0].legend(loc='upper right', fontsize=14, bbox_to_anchor=(1.02, 1.0))
    axs[-1].set_xlabel("Time [s]")

    plt.tight_layout()
    path = '/home/marchnep/Documents/Gitlab_repos/2026_MARCHNER_UQFWI/Fig/fwi_7vp/'
    plt.savefig(path + 'data_UQ.pdf')
    plt.close()


if __name__ == "__main__":
    vp_true = np.array([2700.0, 3200.0, 1900.0, 4200.0, 3800.0, 2200.0, 4500.0])
    set_plot_style()
    res = setup_reference()
    #corner_pdf(res, vp_true)
    #plot_band(res)

    with open('results_svgd.pkl', 'rb') as fp:
        res_load = pickle.load(fp)
        res_svgd, kl_hist = res_load[0], res_load[1]
        print("results loaded !")
    marginals1D_particles(res['samples'], res_svgd, it_num=-1)

    #with open('results_4gvi_mix.pkl', 'rb') as fp:
    #    res_mixgvi = pickle.load(fp)
    #    means, Rs, weights = res_mixgvi[0], res_mixgvi[1], res_mixgvi[2]
    #    print("results mix gvi loaded !")
    #    n_it = len(means)
    
    #target_it = 150
    #it_weights = weights[target_it]
    #print(it_weights)
    #comp_idx = np.random.choice(len(it_weights), size=25000, p=it_weights)
    
    #mog_samples = np.zeros((25000, len(vp_true)))
    #for k in range(len(it_weights)):
    #    m = (comp_idx == k)
    #    if np.sum(m) > 0:
    #        mog_samples[m] = means[target_it][k] + np.random.normal(size=(np.sum(m), len(vp_true))) @ Rs[target_it][k].T
            
    #run_diagnostics(res, mog_samples, method_name=f"MoG VI (it={target_it})")

    #corner_plot_mog(means, Rs, weights, vp_true, it_num=target_it, n_samples=50000)
    #marginals1D_mog(res['samples'], means, Rs, weights, it_num=target_it)