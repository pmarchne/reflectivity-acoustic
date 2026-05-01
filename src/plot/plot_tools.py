import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, ListedColormap, hsv_to_rgb
from src.layers import to_arrays


def set_plot_style():
    plt.rcParams.update(
        {
            "text.usetex": True,  # Use LaTeX
            "font.family": "serif",  # Use serif fonts
            "font.size": 18,  # Set font size
            "axes.labelsize": 18,  # Label font size
            "legend.fontsize": 18,  # Legend font size
            "xtick.labelsize": 18,  # X-axis tick size
            "ytick.labelsize": 18,  # Y-axis tick size
            "figure.figsize": (8, 6),  # Set figure size
            "lines.linewidth": 1.5,  # Line thickness
            "grid.alpha": 0.5,  # Grid transparency
            "savefig.dpi": 300,  # High-resolution images
        }
    )


def plot_reflectivity(omegas, thetas, rmap, omega_c, figsize=(12, 6)):
    plt.figure(figsize=figsize)
    plt.subplot(1, 2, 1)
    valmax = np.max(np.abs(rmap))
    if valmax > 2:
        valmax = 3.0
    plt.imshow(
        np.abs(rmap),
        origin="lower",
        extent=[
            180.0 * thetas[0] / np.pi,
            180.0 * thetas[-1] / np.pi,
            np.real(omegas[0]),
            np.real(omegas[-1]),
        ],
        aspect="auto",
        vmin=0.0,
        vmax=valmax,
    )
    plt.axhline(y=omega_c, color='blue', linestyle='--', linewidth=2)
    plt.xlabel(r"$\theta$ (deg)")
    plt.ylabel(r"$\omega$ (rad/s)")

    omega_idx = np.argmin(np.abs(omegas - omega_c))
    r_omega = np.abs(rmap[omega_idx, :])
    plt.subplot(1, 2, 2)
    plt.plot(180.0 * thetas / np.pi, r_omega, linewidth=2, color='blue')

    plt.xlabel(r"$\theta$ (deg)")
    plt.ylabel(r"$|R|$")
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.1 * valmax])
    plt.xlim([0., 180.0 * thetas[-1] / np.pi])
    plt.tight_layout()
    plt.show()


def plot_reflectivity_complex_plane(R_mesh, theta_re, theta_im, title="Reflection Coefficient $R(\\theta)$"):
    """
    Plots the complex plane using domain coloring.
    
    Parameters:
    - R_mesh: 2D complex array of the function values
    - theta_re: 1D array of the real parts (degrees)
    - theta_im: 1D array of the imaginary parts (radians)
    """
    # 1. Extract Phase and Magnitude
    phase = np.angle(R_mesh)
    magnitude = np.abs(R_mesh)
    # 2. HSV Mapping
    # Hue: Phase (0 to 2pi mapped to 0 to 1)
    H = (phase + np.pi) / (2 * np.pi)
    # Saturation: Fixed at 1 for vibrant colors
    S = np.ones_like(H)
    
    # Value: Magnitude mapping
    # Zeros (mag=0) -> Black (V=0)
    # Poles (mag->inf) -> White (V=1)
    # The exponent 0.5 helps compress the dynamic range so poles don't wash out the whole plot
    V = 1 - 1 / (1 + magnitude**0.5)

    # 3. Convert to RGB
    HSV = np.stack((H, S, V), axis=-1)
    RGB = hsv_to_rgb(HSV)

    # 4. Generate Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    
    extent = [theta_re.min(), theta_re.max(), theta_im.min(), theta_im.max()]
    ax.imshow(RGB, extent=extent, origin='lower', aspect='auto', interpolation='bilinear')
    
    ax.set_title(title)
    ax.set_xlabel(r"Re($\theta$)")
    ax.set_ylabel(r"Im($\theta$)")

    # Add a custom colorbar for Phase
    norm = Normalize(0, 2*np.pi)
    sm = plt.cm.ScalarMappable(cmap='hsv', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, ticks=[0, np.pi, 2*np.pi])
    cbar.set_ticklabels(['0', '$\pi$', '$2\pi$'])
    cbar.set_label('Phase (Hue)')

    plt.tight_layout()
    return fig, ax


def plot_signal_time_freq(time, source_time, freq, source_freq, figsize=(10, 5)):
    plt.figure(figsize=figsize)
    plt.subplot(1, 2, 1)
    plt.plot(time, source_time, "b-")
    plt.xlabel("Time (s)")
    plt.ylabel("Source")
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(freq, np.real(source_freq), "g-", label="real")
    plt.plot(freq, np.imag(source_freq), "r-", label="imag")
    plt.plot(freq, np.abs(source_freq), "k:", label="abs")
    plt.xlabel("Frequency (Hz)")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_1d_profile(layers, param="vp"):
    """
    Plot a 1D depth profile of velocity or density.
    Returns:
        fig, ax
    """
    hs, vps, rhos = to_arrays(layers)
    param_map = {"vp": vps, "rho": rhos}
    param_vals = param_map[param]
    # Calculate depths
    z_interfaces = np.concatenate([[0], np.cumsum(hs)])
    # Create step function for plotting
    z_plot = []
    param_plot = []

    for i in range(len(layers)):
        z_plot.extend([z_interfaces[i], z_interfaces[i + 1]])
        param_plot.extend([param_vals[i], param_vals[i]])

    plt.figure(figsize=(4, 6))
    plt.plot(param_plot, z_plot, color="blue", linewidth=1.5)
    plt.ylabel("Depth z (m)", fontsize=11)
    plt.xlabel(get_param_label(param), fontsize=11)
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.show()


def get_param_label(param):
    """Get formatted label for parameter."""
    labels = {
        "vp": "P-wave velocity (m/s)",
        "rho": "Density (kg/m³)",
        "h": "Thickness (m)",
    }
    return labels.get(param, param)


def plot_layered_config(
    layers, xrecvs=None, zrecvs=None, xs=None, param="vp", cmap="cividis"
):
    """
    Plot a 2D x-z cross-section of layered model showing velocity or density.
    """
    # Convert layers to arrays
    hs, vps, rhos = to_arrays(layers)
    # Select parameter to plot
    param_map = {"vp": vps, "rho": rhos}
    if param not in param_map:
        raise ValueError(f"param must be one of {list(param_map.keys())}")
    param_vals = param_map[param]

    xrecvs = np.asarray(xrecvs)
    x_min, x_max = np.min(xrecvs), np.max(xrecvs)

    # Calculate cumulative depths
    z_interfaces = np.concatenate([[0], np.cumsum(hs)])
    z_max = z_interfaces[-1]

    fig, ax = plt.subplots(figsize=(5, 5))

    # Create filled rectangles for each layer
    cmap_obj = plt.get_cmap(cmap)
    norm = Normalize(vmin=param_vals.min(), vmax=param_vals.max())

    for i in range(len(layers)):
        z_top = z_interfaces[i]
        z_bottom = z_interfaces[i + 1]
        color = cmap_obj(norm(param_vals[i]))

        # Draw rectangle for this layer
        ax.fill_between(
            [0.0, x_max], z_top, z_bottom, color=color, edgecolor="black", linewidth=0.5
        )

    for indr, _ in enumerate(xrecvs):
        ax.plot(xrecvs[indr], zrecvs, "gx")

    ax.plot(xs[0], xs[1], "ro", markersize=8)
    # Set labels and limits
    ax.set_xlabel("x (m)")
    ax.set_ylabel("Depth (m)")
    ax.set_xlim(0.0, x_max)
    ax.set_ylim(0, z_max)
    ax.invert_yaxis()  # depth increases downward

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, label=get_param_label(param))
    plt.tight_layout()
    # plt.show()


def plot_seismogram(
    seismogram,
    xrecvs,
    time,
    vmin=-0.06,
    vmax=0.06,
    cmap="seismic",
    ncolors=256,
    figsize=(8, 10),
):
    """Plot seismogram as image: receivers on x-axis, time on y-axis."""

    seismogram = np.asarray(seismogram)
    if seismogram.shape[0] == len(xrecvs):  # transpose if needed
        seismogram = seismogram.T

    # Discrete colormap
    colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, ncolors))
    cmap_discrete = ListedColormap(colors)

    extent = [np.min(xrecvs), np.max(xrecvs), time[-1], time[0]]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        seismogram,
        aspect="auto",
        origin="upper",
        extent=extent,
        cmap=cmap_discrete,
        vmin=vmin,
        vmax=vmax,
    )

    fig.colorbar(im, ax=ax, label="Amplitude")
    ax.set_xlabel("Receiver position [m]")
    ax.set_ylabel("Time [s]")
    ax.set_title("Seismogram")
    plt.tight_layout()
    plt.show()


def plot_wiggle_traces(
    seismogram, xrecvs, time, trace_step=5, scale=1.0, figsize=(10, 8)
):
    """Plot seismogram as wiggle traces with positive fill."""

    seismogram = np.asarray(seismogram)
    if seismogram.shape[0] == len(xrecvs):
        seismogram = seismogram.T
    fig, ax = plt.subplots(figsize=figsize)
    for i in range(0, len(xrecvs), trace_step):
        trace = seismogram[:, i]
        x_trace = i + trace * scale
        ax.plot(x_trace, time, "k-", linewidth=0.5)
        ax.fill_betweenx(time, i, x_trace, where=(trace > 0), color="black", alpha=0.5)

    ax.set_ylabel("Time [s]")
    ax.set_xlabel("Trace number")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()


def create_plot(X, Y, Z, vp_ref1, vp_ref2, vmin=1000.0, vmax=6000.0, title="my_title"):
    plt.figure(figsize=(8, 5))
    # maxZ = np.max(Z)
    contour_lines = plt.contour(
        X, Y, Z, levels=16, colors="black", linewidths=1.0, linestyles="dotted"
    )
    plt.clabel(
        contour_lines, inline=True, fontsize=6, fmt="%.2f"
    )  # Add isovalue labels
    plt.contourf(X, Y, Z, levels=16, cmap="viridis_r")
    # plt.colorbar(label="$L^2$ misfit", aspect=50)
    plt.colorbar(aspect=50)
    plt.scatter(vp_ref1, vp_ref2, s=115, c="red", marker="*", alpha=1, edgecolors="k")
    plt.xlabel(r"$V_{P,1}$ [m/s]")
    plt.ylabel(r"$V_{P,2}$ [m/s]")
    plt.xlim([vmin, vmax])
    plt.ylim([vmin, vmax])
    plt.title(title)
    plt.tight_layout()
    # plt.show()
