from .find_internal_nodes import find_internal_nodes
from .find_dsbcs import find_dsbcs
from src.norms import norm_type

import numba as nb


@nb.njit(nb.types.Tuple((
        nb.float64,
        nb.int64[:],
        nb.float64[:]))(nb.float64[:, :],
                        nb.float64[:],
                        nb.int64[:],
                        norm_type),
        cache=True)
def prepare_data(X_cluster, core_dists, index, norm):
    row, col, weights, columns = find_internal_nodes(X_cluster, core_dists, norm)
    dsbcs = find_dsbcs(columns, core_dists, row, col, weights)
    return dsbcs, index[columns], core_dists[columns]