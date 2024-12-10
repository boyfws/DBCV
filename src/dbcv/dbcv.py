from src.dbcv.utils import (calculate_core_dist,
                            prepare_data,
                            find_min_dspcs)

import numpy as np

from numba.typed import Dict
from numba import float64, njit, int32, int64

from ..config import norm_type


@njit(float64(
    float64[:, :],
    int32[:],
    int32[:],
    int32[:],
    norm_type),
    cache=True)
def dbcv(X, labels, unique_el, counts_for_uniq, norm):
    dsbcs = np.zeros(shape=unique_el.size,
                     dtype=np.float64)

    internal_obj = Dict.empty(
        key_type=int32,
        value_type=int64[:],
    )

    internal_core_dist = Dict.empty(
        key_type=int32,
        value_type=float64[:],
    )


    min_dspcs = np.full(shape=unique_el.size,
                        fill_value=np.inf,
                        dtype=np.float64)

    for i in range(unique_el.size):
        index = np.flatnonzero(labels == unique_el[i]).astype(np.int64)

        core_dists = calculate_core_dist(X[index], norm)

        dsbcs[i], internal_obj[i], internal_core_dist[i] = prepare_data(X[index],
                                                                        core_dists,
                                                                        index,
                                                                        norm)

    for i in range(unique_el.size):
        for j in range(i + 1, unique_el.size):
            dspc_ij = find_min_dspcs(
                X[internal_obj[i]],
                X[internal_obj[j]],
                internal_core_dist[i],
                internal_core_dist[j],
                norm
            )

            min_dspcs[j] = np.minimum(min_dspcs[j], dspc_ij)
            min_dspcs[i] = np.minimum(min_dspcs[i], dspc_ij)

    np.nan_to_num(min_dspcs, copy=False)
    vcs = (min_dspcs - dsbcs) / (1e-12 + np.maximum(min_dspcs, dsbcs))
    np.nan_to_num(vcs, copy=False, nan=0.0)
    return np.sum(vcs * counts_for_uniq)
