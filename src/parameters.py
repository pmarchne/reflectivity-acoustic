import numpy as np

class Parameters:
    """
    Parameters for reflectivity calculations.
    Attributes:
        time : np.ndarray
            Time array from 0 to total_time.
        dt : float
            Time step.
        f0 : float
            Ricker source central frequency.
        nfft : int
            FFT size (must be >= nt).
        nt : int
            Number of time samples.
    """

    def __init__(self, total_time=3.0, nt=1024, f0=15.0, nfft=2048, epsilon=0.0):
        if nfft < nt:
            raise ValueError("nfft should be larger than nt")
        self.nt = nt
        self.nfft = nfft
        self.f0 = f0
        self.total_time = total_time
        self.dt = total_time / (nt-1)
        self.time = np.arange(nt) * self.dt
        self.epsilon = epsilon  # 0.8 * np.log(50.) / total_time  # default damping factor

    def __repr__(self):
        return (f"Parameters(total_time={self.total_time}, nt={self.nt}, "
                f"f0={self.f0}, nfft={self.nfft}, dt={self.dt}, epsilon={self.epsilon})")

    def create_frequencies(self):
        """
        Create frequency array with complex damping.
        
        Args:
            epsilon: damping factor.
        
        Returns:
            omegas: array of complex angular frequencies.
        """
        freqs = np.fft.rfftfreq(self.nfft, self.dt)
        omegas = 2.0 * np.pi * freqs + 1j * self.epsilon
        print("max recommended epsilon =", np.log(50)/self.total_time)
        return omegas
