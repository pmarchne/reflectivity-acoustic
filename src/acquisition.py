import numpy as np

class Acquisition:
    """Class for acquisition geometry: sources and receivers positions."""
    def __init__(self, sources=None, receivers=None):
        self.sources = np.array(sources) if sources is not None else np.empty((0, 2))
        self.receivers = np.array(receivers) if receivers is not None else np.empty((0, 2))

    @property
    def xs(self):
        return self.sources[:, 0] if len(self.sources) else np.array([])

    @property
    def zs(self):
        return self.sources[:, 1] if len(self.sources) else np.array([])

    @property
    def xr(self):
        return self.receivers[:, 0] if len(self.receivers) else np.array([])

    @property
    def zr(self):
        return self.receivers[:, 1] if len(self.receivers) else np.array([])


if __name__ == "__main__":
    # Example usage, values in meters
    # x and z positions of sources
    sources = [(80.0, 76.0), (150.0, 76.0), (250.0, 76.0), (350.0, 76.0), (480.0, 76.0)]
    Nr = 15
    # x and z positions of receivers
    x_receivers = np.linspace(0.0, 700.0, Nr)
    receivers = [(x, 76.0) for x in x_receivers]

    acq = Acquisition(sources, receivers)

    print("x-sources [m]", acq.xs)
    print("z-sources [m]", acq.zs)
    print("x-receivers [m]", acq.xr)
    print("z-receivers [m]", acq.zr)
