import numpy as np

class Parameters:
    """
    Parameters for reflectivity calculations.
    Attributes:
        time : np.ndarray
            Time array from 0 to total_time.
        dt : float
            Time step.
        freq : float
            Characteristic frequency.
        nfft : int
            FFT size (must be >= nt).
        nt : int
            Number of time samples.
    """

    def __init__(self, total_time=3.0, nt=1024, freq=15.0, nfft=2048):
        if nfft < nt:
            raise ValueError("nfft should be larger than nt")
        self.nt = nt
        self.nfft = nfft
        self.freq = freq
        self.time = np.linspace(0.0, total_time, nt)
        self.total_time = total_time
        self.dt = self.time[1] - self.time[0]
