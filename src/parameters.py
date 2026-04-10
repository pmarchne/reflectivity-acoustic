import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class Parameters:
    dt: float
    nt: int
    nfft: int
    time: np.ndarray
    omegas: np.ndarray
