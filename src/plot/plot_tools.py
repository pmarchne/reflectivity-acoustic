import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "text.usetex": True,         # Use LaTeX
    "font.family": "serif",      # Use serif fonts
    "font.size": 18,             # Set font size
    "axes.labelsize": 18,        # Label font size
    "legend.fontsize": 18,       # Legend font size
    "xtick.labelsize": 18,       # X-axis tick size
    "ytick.labelsize": 18,       # Y-axis tick size
    "figure.figsize": (8,6),     # Set figure size
    "lines.linewidth": 1.5,        # Line thickness
    "grid.alpha": 0.5,           # Grid transparency
    "savefig.dpi": 300           # High-resolution images
})


def plot_reflectivity(omegas, thetas, Rmap, omega_c):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(np.abs(Rmap), origin='lower',
            extent=[180.*thetas[0]/np.pi, 180.*thetas[-1]/np.pi, np.real(omegas[0]), np.real(omegas[-1])],
            aspect='auto',
            vmin=0, vmax=1.0)
    plt.xlabel('Angle (deg)')
    plt.ylabel('frequency (Hz)')
    plt.colorbar(label='|R|')

    omega_idx = np.argmin(np.abs(omegas - omega_c))
    R_at_omega = np.abs(Rmap[omega_idx, :])
    plt.subplot(1, 2, 2)
    plt.plot(180.*thetas/np.pi, R_at_omega, linewidth=2)

    plt.xlabel('Angle (deg)')
    plt.ylabel('|R|')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 2.0])
    plt.tight_layout()
    plt.show()

def plot_signal_time_freq(time, source_time, freq, source_freq):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(time, source_time, 'b-')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('time-domain')
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(freq, np.real(source_freq), 'g-', label="real")
    plt.plot(freq, np.imag(source_freq), 'r-', label="imag")
    plt.plot(freq, np.abs(source_freq), 'k:', label="abs")
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('frequency-domain')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()
