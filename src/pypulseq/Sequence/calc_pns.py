from types import SimpleNamespace
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from pypulseq import Sequence
from pypulseq.utils.safe_pns_prediction import _safe_gwf_to_pns_chunk, safe_plot
from pypulseq.utils.siemens.asc_to_hw import asc_to_hw
from pypulseq.utils.siemens.readasc import readasc

# Number of gradient raster samples processed per SAFE-model chunk (0.3 s on the 10 us raster).
_PNS_CHUNK_SAMPLES = 30_000


def calc_pns(
    obj: Sequence, hardware: SimpleNamespace, time_range: Union[List[float], None] = None, do_plots: bool = True
) -> Tuple[bool, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate PNS using safe model implementation by Szczepankiewicz and Witzel
    See http://github.com/filip-szczepankiewicz/safe_pns_prediction

    Returns pns levels due to respective axes (normalized to 1 and not to 100#)

    Parameters
    ----------
    hardware : SimpleNamespace
        Hardware specifications. See safe_example_hw() from
        the safe_pns_prediction package. Alternatively a text file
        in the .asc format (Siemens) can be passed, e.g. for Prisma
        it is MP_GPA_K2309_2250V_951A_AS82.asc (we leave it as an
        exercise to the interested user to find were these files
        can be acquired from)
    do_plots : bool, optional
        Plot the results from the PNS calculations. The default is True.

    Returns
    -------
    ok : bool
        Boolean flag indicating whether peak PNS is within acceptable limits
    pns_norm : numpy.array [N]
        PNS norm over all gradient channels, normalized to 1
    pns_components : numpy.array [Nx3]
        PNS levels per gradient channel
    t_pns : np.array [N]
        Time axis for the pns_norm and pns_components arrays
    """
    dt = obj.grad_raster_time
    # Get gradients as piecewise-polynomials
    gw_pp = obj.get_gradients(time_range=time_range)
    ng = len(gw_pp)
    max_t = max(g.x[-1] for g in gw_pp if g is not None) - 1e-10

    # Determine sampling points
    if time_range is None:
        nt = int(np.ceil(max_t / dt))
        t = (np.arange(nt) + 0.5) * dt
    else:
        tmax = min(time_range[1], max_t) - max(time_range[0], 0)
        nt = int(np.ceil(tmax / dt))
        t = max(time_range[0], 0) + (np.arange(nt) + 0.5) * dt

    if do_plots:
        plt.figure()
        for i in range(ng):
            if gw_pp[i] is not None:
                plt.plot(gw_pp[i].x[1:-1], gw_pp[i].c[1, :-1])
        plt.title('gradient wave form, in Hz/m')

    if isinstance(hardware, str):
        # this loads the parameters from the provided text file
        asc, _ = readasc(hardware)
        hardware = asc_to_hw(asc)

    # Sample the gradients and run the SAFE model in chunks of samples, carrying the model state
    # (last gradient sample and filter state) from one chunk to the next. This bounds peak memory
    # to the returned arrays plus one chunk, instead of requiring the whole sequence at once.
    n = t.shape[0]
    pns_comp = np.empty((n, 3))
    pns_norm = np.empty(n)
    state = None
    for start in range(0, n, _PNS_CHUNK_SAMPLES):
        stop = min(start + _PNS_CHUNK_SAMPLES, n)
        t_chunk = t[start:stop]

        # Sample gradients
        gw = np.zeros((t_chunk.shape[0], ng))
        for i in range(ng):
            if gw_pp[i] is not None:
                gw[:, i] = gw_pp[i](t_chunk)

        # use the Szczepankiewicz' and Witzel's implementation
        pns_chunk, state = _safe_gwf_to_pns_chunk(gw / obj.system.gamma, obj.grad_raster_time, hardware, state)

        pns_comp[start:stop] = 0.01 * pns_chunk
        pns_norm[start:stop] = np.sqrt((pns_comp[start:stop] ** 2).sum(axis=1))

    # calc the final ok/not_ok
    ok = bool(np.all(pns_norm < 1))

    # ready
    if do_plots:
        # plot results
        plt.figure()
        safe_plot(pns_comp * 100, obj.grad_raster_time)

    return ok, pns_norm, pns_comp, t
