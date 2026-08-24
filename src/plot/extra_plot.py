import matplotlib.pyplot as plt
import numpy as np
from src.plot.plot_tools import set_plot_style

set_plot_style()
path = '' #'/home/marchnep/Documents/Gitlab_repos/2026_MARCHNER_UQFWI/Fig/fwi_7vp/'

data = np.load('swd_convergence_seed16_fine.npz')
mixgvi_k5_swd = data['SWD5']
mixgvi_k10_swd = data['SWD10']
svgd_swd = data['SWD_S']

step = 6
iterations = np.arange(len(svgd_swd))[::step]
svgd_downsampled = svgd_swd[::step] 
k5_downsampled = mixgvi_k5_swd[::step] 
k10_downsampled = mixgvi_k10_swd[::step] 

plt.figure(figsize=(7, 3.7), dpi=300)
plt.plot(
    iterations,
    svgd_downsampled,
    label=r'SVGD',
    color='green',
    linewidth=2.25,
    marker='o',
    alpha=0.9,
    markersize=4,
)
plt.plot(
    iterations,
    k5_downsampled,
    label=r'$K=5$',
    color='darkorange',
    linewidth=2.25,
    alpha=0.9,
    marker='s',
    markersize=4,
)
plt.plot(
    iterations,
    k10_downsampled,
    label=f'$K=10$',
    color='red',
    linewidth=2.25,
    alpha=0.9,
    marker='^',
    markersize=4,
)

plt.xlim([0, 199])
plt.xlabel('Iterations')
plt.ylabel('SWD')
plt.legend(frameon=False)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(path+'seed16_SWD_progress.pdf')
#plt.show()


# generic function to extract all data
'''
def plot_multi_seed_metrics(seeds, target_it, res_ref, dim):
    """
    Loops over seeds, loads final iteration results for SVGD, MixGVI-5, and MixGVI-10,
    computes MMD and SWD
    """
    methods = ['SVGD', 'MixGVI (K=5)', 'MixGVI (K=10)']
    metrics_mmd = {m: [] for m in methods}
    metrics_swd = {m: [] for m in methods}
    
    for seed in seeds:
        print("seed nr", seed)
        try:
            with open('res_svgd_unconst_rng'+str(seed)+'.pkl', 'rb') as fp:
                res_svgd, _ = pickle.load(fp)
                svgd_samples = res_svgd[target_it]
                mmd_s, swd_s = run_diagnostics(res_ref, svgd_samples, method_name="SVGD", verbose=False)
                metrics_mmd['SVGD'].append(mmd_s)
                metrics_swd['SVGD'].append(swd_s)
        except FileNotFoundError:
            print(f"SVGD file for seed {seed} not found. Skipping.")

        try:
            with open('test_unconst_MoG5_rng'+str(seed)+'.pkl', 'rb') as fp:
                res_mix5 = pickle.load(fp)
                means5, Rs5, weights5 = res_mix5[0], res_mix5[1], res_mix5[2]
                mog5_samples = get_MoG(means5, Rs5, weights5, target_it, dim)
                mmd_5, swd_5 = run_diagnostics(res_ref, mog5_samples, method_name="MixGVI-5", verbose=False)
                metrics_mmd['MixGVI (K=5)'].append(mmd_5)
                metrics_swd['MixGVI (K=5)'].append(swd_5)
        except FileNotFoundError:
            print(f"MixGVI-5 file for seed {seed} not found. Skipping.")

        try:
            with open('test_unconst_MoG10_rng'+str(seed)+'.pkl', 'rb') as fp:
                res_mix10 = pickle.load(fp)
                means10, Rs10, weights10 = res_mix10[0], res_mix10[1], res_mix10[2]
                mog10_samples = get_MoG(means10, Rs10, weights10, target_it, dim)
                mmd_10, swd_10 = run_diagnostics(res_ref, mog10_samples, method_name="MixGVI-10", verbose=False)
                metrics_mmd['MixGVI (K=10)'].append(mmd_10)
                metrics_swd['MixGVI (K=10)'].append(swd_10)
        except FileNotFoundError:
            print(f"MixGVI-10 file for seed {seed} not found. Skipping.")

    x_pos = np.arange(len(methods))
    mmd_means = [np.mean(metrics_mmd[m]) if metrics_mmd[m] else 0 for m in methods]
    mmd_stds  = [np.std(metrics_mmd[m]) if metrics_mmd[m] else 0 for m in methods]
    swd_means = [np.mean(metrics_swd[m]) if metrics_swd[m] else 0 for m in methods]
    swd_stds  = [np.std(metrics_swd[m]) if metrics_swd[m] else 0 for m in methods]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # MMD Plot
    axes[0].bar(x_pos, mmd_means, yerr=mmd_stds, align='center', alpha=0.8, capsize=3, color=['#ff7f0e', '#1f77b4', '#2ca02c'])
    axes[0].set_ylabel('Joint MMD')
    axes[0].set_title(f'Comparison of MMD across 10 Seeds (It={target_it})')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(methods)
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)

    # SWD Plot
    axes[1].bar(x_pos, swd_means, yerr=swd_stds, align='center', alpha=0.8, capsize=3, color=['#ff7f0e', '#1f77b4', '#2ca02c'])
    axes[1].set_ylabel('Joint SWD')
    axes[1].set_title(f'Comparison of SWD across 10 Seeds (It={target_it})')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(methods)
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()
'''