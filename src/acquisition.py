import numpy as np


class Acquisition:
    """Acquisition geometry: sources and receivers."""

    def __init__(self, sources=None, receivers=None):
        self.sources = self._to_points_array(sources)
        self.receivers = self._to_points_array(receivers)

    @staticmethod
    def _to_points_array(points):
        if points is None:
            return np.empty((0, 2), dtype=float)
        arr = np.asarray(points, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] != 2:
            raise ValueError("Points must have shape (N, 2)")
        return arr

    @property
    def xs(self):
        return self.sources[:, 0]

    @property
    def zs(self):
        return self.sources[:, 1]

    @property
    def xr(self):
        return self.receivers[:, 0]

    @property
    def zr(self):
        return self.receivers[:, 1]

    def distances_direct(self):
        dx = self.sources[:, 0][:, None] - self.receivers[:, 0][None, :]
        dz = self.sources[:, 1][:, None] - self.receivers[:, 1][None, :]
        return np.sqrt(dx**2 + dz**2).ravel()

    def distances_ghost(self):
        dx = self.sources[:, 0][:, None] - self.receivers[:, 0][None, :]
        # For the ghost source, the vertical distance is the sum of source and receiver depths
        dz = self.sources[:, 1][:, None] + self.receivers[:, 1][None, :]
        return np.sqrt(dx**2 + dz**2).ravel()
