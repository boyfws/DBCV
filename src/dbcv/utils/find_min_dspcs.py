from ...config import norm_type
import numba as nb

# Dtypes
from numpy import inf, float64

# Other
from numpy import full, min, maximum


@nb.njit(nb.float64(
        nb.float64[:, :],
        nb.float64[:, :],
        nb.float64[:],
        nb.float64[:],
        norm_type),

    cache=True,
    parallel=True
    )
def find_min_dspcs(X_cluster_i, X_cluster_j, core_dists_i, core_dists_j, norm):
    if X_cluster_i.size == 0 or X_cluster_j.size == 0:
        return inf

    min_values = full(X_cluster_i.size, inf, dtype=float64)
    for k in nb.prange(X_cluster_i.size):
        L = norm(X_cluster_j, X_cluster_i[k])
        L = maximum(L, core_dists_j)
        L = maximum(L, core_dists_i[k])
        min_values[k] = min(L)

    return min(min_values)



