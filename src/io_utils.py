import numpy as np
from pathlib import Path


def read_fd_observations(
    file_path: Path,
    nr: int,
    nt_ref: int,
    nt_cal: int,
    total_time: float,
    ind_traces: list,
    normalize: bool = True,
):
    """
    Reads FD binary data, selects specific traces, and resamples to calculation time grid.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Missing observation file: {file_path}")

    # Read binary in float32
    raw = np.fromfile(file_path, dtype=np.float32)

    if raw.size != nr * nt_ref:
        raise ValueError(f"File size mismatch. Got {raw.size}, expected {nr * nt_ref}")

    # Reshape and select traces: (nt_ref, nr) -> (nt_ref, selected_nr)
    seismo = raw.reshape((nr, nt_ref)).T.astype(float)
    seismo_selected = seismo[:, ind_traces]

    # Time axes for interpolation
    t_ref = np.linspace(0, total_time, nt_ref)
    t_cal = np.linspace(0, total_time, nt_cal)

    # Vectorized interpolation across selected traces
    resampled_data = np.apply_along_axis(
        lambda x: np.interp(t_cal, t_ref, x), 0, seismo_selected
    ).T

    scale = 1.0
    if normalize:
        scale = np.max(np.abs(resampled_data))
        if scale > 0:
            resampled_data /= scale

    return resampled_data, scale
