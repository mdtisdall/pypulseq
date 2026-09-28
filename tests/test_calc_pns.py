import importlib
from typing import List, Tuple, Union

import numpy as np
import pypulseq as pp
import pytest
from pypulseq import Sequence
from pypulseq.utils.safe_pns_prediction import safe_example_hw, safe_gwf_to_pns

SYSTEM = pp.Opts(max_grad=28, grad_unit='mT/m', max_slew=150, slew_unit='T/m/s')


def whole_sequence_pns(
    seq: Sequence, hardware, time_range: Union[List[float], None] = None
) -> Tuple[bool, np.ndarray, np.ndarray, np.ndarray]:
    # This is the whole-sequence computation of calc_pns, before the change to chunks: a copy of
    # calc_pns's body, minus the plotting and the .asc-file string path for `hardware`.
    dt = seq.grad_raster_time
    gw_pp = seq.get_gradients(time_range=time_range)
    ng = len(gw_pp)
    max_t = max(g.x[-1] for g in gw_pp if g is not None) - 1e-10

    if time_range is None:
        nt = int(np.ceil(max_t / dt))
        t = (np.arange(nt) + 0.5) * dt
    else:
        tmax = min(time_range[1], max_t) - max(time_range[0], 0)
        nt = int(np.ceil(tmax / dt))
        t = max(time_range[0], 0) + (np.arange(nt) + 0.5) * dt

    gw = np.zeros((t.shape[0], ng))
    for i in range(ng):
        if gw_pp[i] is not None:
            gw[:, i] = gw_pp[i](t)

    [pns_comp, res] = safe_gwf_to_pns(
        gw / seq.system.gamma, np.nan * np.ones(t.shape[0]), seq.grad_raster_time, hardware
    )

    pns_comp = 0.01 * pns_comp[~np.isfinite(res.rf[1:]), :]

    pns_norm = np.sqrt((pns_comp**2).sum(axis=1))
    ok = all(pns_norm < 1)

    return ok, pns_norm, pns_comp, t


def build_dense_sequence() -> Sequence:
    # A trapezoid on each axis every 10 ms, for 3 s (300 blocks, about 300,000 samples).
    seq = Sequence(system=SYSTEM)
    gx = pp.make_trapezoid('x', area=1000, system=SYSTEM)
    gy = pp.make_trapezoid('y', area=1000, system=SYSTEM)
    gz = pp.make_trapezoid('z', area=1000, system=SYSTEM)
    delay = pp.make_delay(10e-3)
    for _ in range(300):
        seq.add_block(gx, gy, gz, delay)
    return seq


def build_one_axis_sequence() -> Sequence:
    # The same pattern, x only, for 1 s (100 blocks, about 100,000 samples).
    seq = Sequence(system=SYSTEM)
    gx = pp.make_trapezoid('x', area=1000, system=SYSTEM)
    delay = pp.make_delay(10e-3)
    for _ in range(100):
        seq.add_block(gx, delay)
    return seq


def build_gap_sequence() -> Sequence:
    # One block of trapezoids, a 0.5 s delay block, then one more block of trapezoids.
    seq = Sequence(system=SYSTEM)
    gx = pp.make_trapezoid('x', area=1000, system=SYSTEM)
    gy = pp.make_trapezoid('y', area=1000, system=SYSTEM)
    gz = pp.make_trapezoid('z', area=1000, system=SYSTEM)
    seq.add_block(gx, gy, gz)
    seq.add_block(pp.make_delay(0.5))
    seq.add_block(gx, gy, gz)
    return seq


def build_arbitrary_sequence() -> Sequence:
    # A smooth arbitrary gradient on y (a sine lobe within the system limits), repeated in
    # 2 ms blocks for about 0.5 s.
    seq = Sequence(system=SYSTEM)
    dt = SYSTEM.grad_raster_time
    duration = 2e-3
    n = round(duration / dt)
    t = (np.arange(n) + 0.5) * dt
    waveform = 0.5 * SYSTEM.max_grad * np.sin(np.pi * t / duration)
    grad = pp.make_arbitrary_grad('y', waveform, system=SYSTEM)
    for _ in range(250):
        seq.add_block(grad)
    return seq


def build_short_sequence() -> Sequence:
    # A single block of trapezoids: far less than 100,000 samples.
    seq = Sequence(system=SYSTEM)
    gx = pp.make_trapezoid('x', area=1000, system=SYSTEM)
    gy = pp.make_trapezoid('y', area=1000, system=SYSTEM)
    gz = pp.make_trapezoid('z', area=1000, system=SYSTEM)
    seq.add_block(gx, gy, gz)
    return seq


@pytest.fixture(scope='module')
def hw():
    return safe_example_hw()


@pytest.fixture(scope='module')
def dense_seq():
    return build_dense_sequence()


@pytest.fixture(scope='module')
def one_axis_seq():
    return build_one_axis_sequence()


@pytest.fixture(scope='module')
def gap_seq():
    return build_gap_sequence()


@pytest.fixture(scope='module')
def arbitrary_seq():
    return build_arbitrary_sequence()


@pytest.fixture(scope='module')
def short_seq():
    return build_short_sequence()


def assert_same_pns(actual, expected):
    ok, pns_norm, pns_comp, t = actual
    ok_ref, pns_norm_ref, pns_comp_ref, t_ref = expected
    assert type(ok) is bool
    assert ok == ok_ref
    np.testing.assert_array_equal(pns_norm, pns_norm_ref)
    np.testing.assert_array_equal(pns_comp, pns_comp_ref)
    np.testing.assert_array_equal(t, t_ref)


@pytest.mark.parametrize('seq_fixture', ['dense_seq', 'one_axis_seq', 'gap_seq', 'arbitrary_seq', 'short_seq'])
def test_calculate_pns_matches_whole_sequence_pns(request, seq_fixture, hw):
    seq = request.getfixturevalue(seq_fixture)
    actual = seq.calculate_pns(hw, do_plots=False)
    expected = whole_sequence_pns(seq, hw)
    assert_same_pns(actual, expected)


@pytest.mark.parametrize('chunk_size', [997, 30_000, 100_000])
def test_calculate_pns_matches_whole_sequence_pns_with_time_range(monkeypatch, dense_seq, hw, chunk_size):
    calc_pns_module = importlib.import_module('pypulseq.Sequence.calc_pns')
    monkeypatch.setattr(calc_pns_module, '_PNS_CHUNK_SAMPLES', chunk_size, raising=False)
    time_range = [0.5002, 1.5]
    actual = dense_seq.calculate_pns(hw, time_range=time_range, do_plots=False)
    expected = whole_sequence_pns(dense_seq, hw, time_range=time_range)
    assert_same_pns(actual, expected)


@pytest.mark.parametrize('seq_fixture', ['dense_seq', 'gap_seq'])
@pytest.mark.parametrize('chunk_frac', ['997', 'n_half', 'n_minus_1', 'n', 'n_plus_1'])
def test_calculate_pns_matches_whole_sequence_pns_with_chunking(request, monkeypatch, seq_fixture, chunk_frac, hw):
    # calc_pns computes the SAFE model in chunks of _PNS_CHUNK_SAMPLES samples. Chunk ends then
    # fall at different points of the gradients. raising=False lets the test also run on the
    # code before the chunks.
    seq = request.getfixturevalue(seq_fixture)
    expected = whole_sequence_pns(seq, hw)
    n = len(expected[3])
    chunk_sizes = {'997': 997, 'n_half': n // 2, 'n_minus_1': n - 1, 'n': n, 'n_plus_1': n + 1}
    calc_pns_module = importlib.import_module('pypulseq.Sequence.calc_pns')
    monkeypatch.setattr(calc_pns_module, '_PNS_CHUNK_SAMPLES', chunk_sizes[chunk_frac], raising=False)

    actual = seq.calculate_pns(hw, do_plots=False)
    assert_same_pns(actual, expected)
