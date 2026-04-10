from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # Forward modeling
    total_time: float = 2.0
    f0: float = 5.0
    epsilon: float = 0.5

    # Acquisition
    n_receivers: int = 24
    x_min: float = 100.0
    x_max: float = 4000.0
    x_src: float = 30.0
    z_src: float = 76.0
    z_rec: float = 76.0
    # numerics
    nq_prop: int = 256
    nq_evan: int = 128
    nfft_pad_factor: int = 2

    # physics
    free_surface: bool = True
    delay: float = 0.2
    source_deriv: bool = True

    # Noise
    noise_level: float = 0.1
    seed: int | None = None

    # ---------- Validation ----------
    def validate(self):
        if self.total_time <= 0:
            raise ValueError("total_time must be positive")
        if self.f0 <= 0:
            raise ValueError("f0 must be positive")
        if self.n_receivers <= 0:
            raise ValueError("n_receivers must be > 0")
        
