from .prim_mst import prim_mst
from ...config import norm_type

import numba as nb

from numpy import bincount, hstack, array, max, float64



@nb.njit(nb.types.Tuple((
        nb.float64,
        nb.int64[:],
        nb.float64[:]))(nb.float64[:, :],
                        nb.float64[:],
                        nb.int64[:],
                        norm_type),
        cache=True,
        parallel=True)
def prepare_data(X_cluster, core_dists, index, norm):
    row, col, weights = prim_mst(X_cluster, core_dists, norm)

    bit_mask = bincount(hstack((row, col))) > 1

    assert row.shape[0] == col.shape[0] == bit_mask.shape[0], "Unreachable condition reached - please report an Issue"

    dsbcs = max(
        array(
            [
                weights[j] for j in nb.prange(row.shape[0]) if bit_mask[row[j]] * bit_mask[col[j]]
            ],
            dtype=float64)
    )
    return dsbcs, index[bit_mask], core_dists[bit_mask]