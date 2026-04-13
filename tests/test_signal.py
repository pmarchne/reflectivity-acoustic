import numpy as np
from src.utilities import (
    ricker_wavelet,
    source_frequency,
    inverse_fft_signal,
    adjoint_inverse_fft_signal,
)


def test_ricker_spectrum(param_small, config_small):
    s_w = source_frequency(param_small, config_small)
    freqs = param_small.omegas / (2.0 * np.pi)

    exact = (
        (2.0 / np.sqrt(np.pi))
        * freqs**2
        / (config_small.f0**3)
        * np.exp(-(freqs / config_small.f0) ** 2)
        * np.exp(2j * np.pi * freqs * config_small.delay)
    )

    np.testing.assert_allclose(s_w, exact, atol=1e-6, rtol=0.0)


def test_inverse_fft_signal(param_fft, config_fft):
    s_w = source_frequency(param_fft, config_fft)
    s_t = ricker_wavelet(param_fft.time, config_fft.f0, config_fft.delay)
    inv_time = inverse_fft_signal(s_w[None, :], param_fft, config_fft)
    np.testing.assert_allclose(inv_time[0], s_t, atol=1e-2, rtol=0.0)


def test_adjoint_inverse_fft_signal(param_fft, config_fft):
    rng = np.random.default_rng(0)
    Ntr = 6
    Nw = param_fft.nfft // 2 + 1
    Nt = param_fft.nt

    x = rng.standard_normal((Ntr, Nw)) + 1j * rng.standard_normal((Ntr, Nw))
    y = rng.standard_normal((Ntr, Nt))

    Fx = inverse_fft_signal(x, param_fft, config_fft)
    Fty = adjoint_inverse_fft_signal(y, param_fft, config_fft)

    lhs = np.real(np.vdot(Fx, y))
    rhs = np.real(np.vdot(x, Fty))

    rel_err = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-16)
    print("rel_err:", rel_err)
    assert rel_err < 1e-6
