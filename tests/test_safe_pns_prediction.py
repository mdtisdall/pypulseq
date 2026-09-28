import copy
from types import SimpleNamespace

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


def split_into_sizes(gwf, sizes):
    # Split gwf (n, 3) into consecutive chunks of the given lengths, which must sum to n.
    assert sum(sizes) == gwf.shape[0]
    chunks = []
    start = 0
    for size in sizes:
        chunks.append(gwf[start : start + size])
        start += size
    return chunks


def uniform_chunk_sizes(n, size):
    # A list of chunk lengths of the given size, with a shorter final chunk if size
    # does not evenly divide n.
    sizes = [size] * (n // size)
    remainder = n - sum(sizes)
    if remainder:
        sizes.append(remainder)
    return sizes


def run_in_chunks(gwf, dt, hw, sizes):
    # Call _safe_gwf_to_pns_chunk once per chunk, in order, threading the state from one
    # call into the next, and concatenate the returned pns chunks.
    state = None
    pns_chunks = []
    for chunk in split_into_sizes(gwf, sizes):
        pns_chunk, state = safe_pns_prediction._safe_gwf_to_pns_chunk(chunk, dt, hw, state)
        pns_chunks.append(pns_chunk)
    return np.concatenate(pns_chunks, axis=0), state


def reference_pns(gwf, dt, hw):
    # The rows safe_gwf_to_pns produces for the real (non-padded) samples, as
    # calc_pns.py selects them.
    n = gwf.shape[0]
    pns_full, res = safe_gwf_to_pns(gwf, np.nan * np.ones(n), dt, hw)
    return pns_full[~np.isfinite(res.rf[1:]), :]


def assert_chunking_matches_the_full_computation(gwf, dt, hw, sizes):
    chunked, _ = run_in_chunks(gwf, dt, hw, sizes)
    np.testing.assert_array_equal(chunked, reference_pns(gwf, dt, hw))


@pytest.mark.parametrize('chunk_size', [1, 7, 76])
def test_chunking_the_example_waveform_matches_the_full_computation(chunk_size):
    gwf, _, dt = safe_example_gwf()
    hw = safe_example_hw()
    assert_chunking_matches_the_full_computation(gwf, dt, hw, uniform_chunk_sizes(gwf.shape[0], chunk_size))


def random_gwf_on_the_10us_raster():
    return np.cumsum(np.random.default_rng(0).normal(size=(20000, 3)), axis=0) * 1e-5


@pytest.mark.parametrize('chunk_size', [7, 1000, 4096, 20000])
def test_chunking_a_longer_waveform_matches_the_full_computation(chunk_size):
    gwf = random_gwf_on_the_10us_raster()
    dt = 1e-5
    hw = safe_example_hw()
    assert_chunking_matches_the_full_computation(gwf, dt, hw, uniform_chunk_sizes(gwf.shape[0], chunk_size))


def test_chunking_with_uneven_sizes_matches_the_full_computation():
    gwf = random_gwf_on_the_10us_raster()
    dt = 1e-5
    hw = safe_example_hw()
    sizes = [1, 999, 3000, gwf.shape[0] - (1 + 999 + 3000)]
    assert_chunking_matches_the_full_computation(gwf, dt, hw, sizes)


def test_chunking_matches_the_full_computation_with_doubled_time_constants():
    gwf, _, dt = safe_example_gwf()
    hw = copy.deepcopy(safe_example_hw())
    for axis in 'xyz':
        hw_ax = getattr(hw, axis)
        hw_ax.tau1 *= 2
        hw_ax.tau2 *= 2
        hw_ax.tau3 *= 2
    assert_chunking_matches_the_full_computation(gwf, dt, hw, uniform_chunk_sizes(gwf.shape[0], 7))


def test_no_state_is_the_same_as_an_explicit_zero_state():
    gwf, _, dt = safe_example_gwf()
    hw = safe_example_hw()
    chunk = gwf[:7]

    pns_none, state_none = safe_pns_prediction._safe_gwf_to_pns_chunk(chunk, dt, hw, None)

    zero_state = SimpleNamespace(g_last=np.zeros(3), zi=np.zeros((3, 3)))
    pns_zero, state_zero = safe_pns_prediction._safe_gwf_to_pns_chunk(chunk, dt, hw, zero_state)

    np.testing.assert_array_equal(pns_none, pns_zero)
    np.testing.assert_array_equal(state_none.g_last, state_zero.g_last)
    np.testing.assert_array_equal(state_none.zi, state_zero.zi)


def test_the_chunk_call_does_not_modify_its_inputs():
    gwf, _, dt = safe_example_gwf()
    hw = safe_example_hw()

    _, state = safe_pns_prediction._safe_gwf_to_pns_chunk(gwf[:7], dt, hw, None)
    state_g_last_before = state.g_last.copy()
    state_zi_before = state.zi.copy()

    chunk = gwf[7:20].copy()
    chunk_before = chunk.copy()

    safe_pns_prediction._safe_gwf_to_pns_chunk(chunk, dt, hw, state)

    np.testing.assert_array_equal(chunk, chunk_before)
    np.testing.assert_array_equal(state.g_last, state_g_last_before)
    np.testing.assert_array_equal(state.zi, state_zi_before)


def test_the_returned_state_holds_the_last_gradient_sample_and_a_3x3_zi():
    gwf, _, dt = safe_example_gwf()
    hw = safe_example_hw()
    chunk = gwf[:7]

    _, state = safe_pns_prediction._safe_gwf_to_pns_chunk(chunk, dt, hw, None)

    np.testing.assert_array_equal(state.g_last, chunk[-1])
    assert state.zi.shape == (3, 3)
