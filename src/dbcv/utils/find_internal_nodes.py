from .prim_mst import prim_mst
from src.norms import norm_type

import numba as nb

from numpy import flatnonzero, bincount, hstack, int32


@nb.njit(nb.types.Tuple((
        nb.int32[:],
        nb.int32[:],
        nb.float64[:],
        nb.int32[:]))(nb.float64[:, :],
                  nb.float64[:],
                  norm_type),
         cache=True
         )
def find_internal_nodes(X_cluster, core_dists, norm):
    row, col, weights = prim_mst(X_cluster, core_dists, norm)
    columns = flatnonzero(
            bincount(
                hstack((row, col))
            ) > 1
    ).astype(int32)
    return row, col, weights, columns
