import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, ListedColormap, hsv_to_rgb
from src.layers import to_arrays

''' A bunch of functions for creating figures '''

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
    V = 1 - 1 / (1 + magnitude**0.5)

    # 3. Convert to RGB
    HSV = np.stack((H, S, V), axis=-1)
    RGB = hsv_to_rgb(HSV)

    # 4. Generate Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    
    theta_re_deg = np.degrees(theta_re)
    theta_im_deg = np.degrees(theta_im)
    extent = [theta_re_deg.min(), theta_re_deg.max(), theta_im_deg.min(), theta_im_deg.max()]
    ax.imshow(RGB, extent=extent, origin='lower', aspect='auto', interpolation='bilinear')
    
    ax.set_title(title)
    ax.set_xlabel(r"Re($\theta$) (deg)")
    ax.set_ylabel(r"Im($\theta$) (deg)")

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


def plot_seismogram(
    seismogram,
    xrecvs,
    time,
    x_off=None,
    vmin=-0.06,
    vmax=0.06,
    cmap="seismic",
    ncolors=256,
    figsize=(8, 10),
):
    """Plot seismogram as image: receivers on x-axis, time on y-axis."""

    seismogram = np.asarray(seismogram)
    if seismogram.shape[0] == len(xrecvs):
        seismogram = seismogram.T

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

    # Add vertical dashed lines
    if x_off is not None:
        for x in x_off:
            ax.axvline(
                x=x,
                color="k",
                linestyle="--",
                linewidth=1.8,
                alpha=0.5,
            )

    fig.colorbar(im, ax=ax)
    ax.set_xlabel("Offset [m]")
    ax.set_ylabel("Time [s]")
    plt.tight_layout()


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


def create_plot(X, Y, Z, vp_ref1, vp_ref2, vmin=1000.0, vmax=6000.0, title=r"normalized $L^2$ misfit"):
    plt.figure(figsize=(8, 5))
    # maxZ = np.max(Z)
    contour_lines = plt.contour(
        X, Y, Z, levels=16, colors="black", linewidths=1.0, linestyles="dotted"
    )
    plt.clabel(
        contour_lines, inline=True, fontsize=6, fmt="%.2f"
    )  # Add isovalue labels
    plt.contourf(X, Y, Z, levels=16, cmap="viridis_r")
    plt.colorbar(label=title, aspect=50)
    plt.colorbar(aspect=50)
    plt.scatter(vp_ref1, vp_ref2, s=115, c="red", marker="*", alpha=1, edgecolors="k")
    plt.xlabel(r"$v_1$ [m/s]")
    plt.ylabel(r"$v_2$ [m/s]")
    plt.xlim([vmin, vmax])
    plt.ylim([vmin, vmax])
    #plt.title(title)
    plt.tight_layout()
    # plt.show()


def plot_post_velocity(layers, post_mean, post_std_mean, post_map):
    z_plot = []
    param_plot = []
    prior_plot = []
    post_plot = []
    m_in = np.array([0., 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0])
    std = 1000.

    #z_int = np.array([0.0, 100.0, 200.0, 275.0, 375.0, 400.0, 500.0, 550.0, 700.0])
    #vp    = np.array([1505.0, 2700.0, 3200.0, 1900.0, 4200.0, 3800.0, 2200.0, 4500.0])
    #rho = np.full_like(vp, 2000.0)
    #layers = create_layers_from_interfaces(z_int, vp, rho)
    
    #post_mean = np.array([0., 2702.4019070178424, 3112.064015550052, 2077.507277017946, 3083.3118095056675, 3705.1231211286636, 2391.3892823343713, 3491.836473134705])
    #post_std_mean = np.array([0., 45.01521047283904, 175.30221476342064, 176.01326592858354, 750.3434919297522, 326.41464119327804, 610.4518581244358, 821.9363398943274])
    #post_map = np.array([0., 2696.8198354246188, 3074.4423873792193, 1980.826105977049, 3202.2884114407298, 3700.4895477382443, 2397.930062650532, 4124.598347538598])

    for i in range(len(layers)):
        z_plot.extend([z_int[i], z_int[i + 1]])
        param_plot.extend([vp[i], vp[i]])
        prior_plot.extend([m_in[i], m_in[i]])
        post_plot.extend([post_mean[i], post_mean[i]])

    plt.figure(figsize=(4, 6))

    # Prior uncertainty
    for i in range(len(layers)):
        plt.fill_betweenx(
            [z_int[i], z_int[i+1]],
            m_in[i] - 1.*std,
            m_in[i] + 1.*std,
            color="gray",
            alpha=0.25,
            linewidth=0,
        )
    # prior mean
    for i in range(len(layers)):
        plt.plot([m_in[i], m_in[i]],
                [z_int[i], z_int[i+1]],
                color="grey",
                linewidth=1.5,
                linestyle='--',
                label="prior mean" if i == 0 else "")


    # Post std
    for i in range(len(layers)):
        plt.fill_betweenx(
            [z_int[i], z_int[i+1]],
            post_mean[i] - post_std_mean[i],
            post_mean[i] + post_std_mean[i],
            color="blue",
            alpha=0.25,
            linewidth=0,
        )
    # post mean
    for i in range(len(layers)):
        plt.plot([post_mean[i], post_mean[i]],
                [z_int[i], z_int[i+1]],
                color="b",
                linewidth=1.8,
                linestyle='-.',
                label="post mean" if i == 0 else "")

    # True model
    plt.plot(param_plot, z_plot, color="red",
            linewidth=2., label="reference")

    plt.ylabel("Depth [m]")
    plt.xlabel(r"$v_P$ [m/s]")
    plt.gca().invert_yaxis()
    plt.grid(alpha=0.4)
    #plt.xticks([2000, 3000, 4000, 5000, 6000], fontsize=14)
    plt.xlim(1000, 6000)
    plt.ylim(700, 0)

    plt.legend(loc='upper right', fontsize=13)
    plt.tight_layout()
    #plt.show()
    #plt.savefig(path_save+'ref_profile.pdf')