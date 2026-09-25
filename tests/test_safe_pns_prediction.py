import numpy as np
import pytest
from pypulseq.utils import safe_pns_prediction
from pypulseq.utils.safe_pns_prediction import safe_example_gwf, safe_example_hw, safe_gwf_to_pns, safe_tau_lowpass

DT_MS = 0.01  # the 10 us gradient raster, in ms as safe_pns_model gives it


def convolution_lowpass(dgdt, tau, dt, eps=1e-16):
    # The filter before the recursion: a convolution with the kernel
    # alpha * (1 - alpha)^k, cut where (1 - alpha)^k is below eps.
    alpha = dt / (tau + dt)
    n = min(round(np.log(eps) / np.log(1 - alpha)), dgdt.shape[0])
    filt = (1 - alpha) ** np.arange(n)
    return alpha * np.convolve(dgdt, filt)[: dgdt.shape[0]]


def assert_same_filter(new, old):
    # The recursion and the convolution add the same terms in a different order, and
    # the convolution cuts its kernel at 1e-16. So allow a relative 1e-12, and an
    # absolute 1e-12 of the largest value for samples near zero.
    np.testing.assert_allclose(new, old, rtol=1e-12, atol=1e-12 * np.max(np.abs(old)))


def example_time_constants():
    hw = safe_example_hw()
    return [
        pytest.param(getattr(getattr(hw, axis), f'tau{i}'), id=f'{axis}.tau{i}') for axis in 'xyz' for i in (1, 2, 3)
    ]


@pytest.mark.parametrize('tau', example_time_constants())
@pytest.mark.parametrize('n', [1, 100, 50_000])
def test_lowpass_equals_the_convolution(tau, n):
    # 50,000 samples are longer than the longest cut kernel (about 11,000 samples).
    x = np.random.default_rng(0).normal(size=n) * 1e3
    for signal in (x, np.abs(x)):
        assert_same_filter(safe_tau_lowpass(signal, tau, DT_MS), convolution_lowpass(signal, tau, DT_MS))


def test_lowpass_keeps_the_shape_and_ignores_eps():
    x = np.random.default_rng(1).normal(size=1000)
    y = safe_tau_lowpass(x, 3.0, DT_MS)
    assert y.shape == x.shape
    np.testing.assert_array_equal(safe_tau_lowpass(x, 3.0, DT_MS, eps=1e-3), y)


def test_pns_of_the_example_waveform_equals_the_convolution(monkeypatch):
    gwf, rf, dt = safe_example_gwf()
    hw = safe_example_hw()
    new, _ = safe_gwf_to_pns(gwf, rf, dt, hw)
    monkeypatch.setattr(safe_pns_prediction, 'safe_tau_lowpass', convolution_lowpass)
    old, _ = safe_gwf_to_pns(gwf, rf, dt, hw)
    assert_same_filter(new, old)
