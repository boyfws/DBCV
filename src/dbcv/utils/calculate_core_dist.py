from src.config import norm_type

import numba as nb

# Types
from numpy import float64, inf

# Array creating
from numpy import zeros

# Other staff
from numpy import sum


@nb.njit(nb.float64[:](nb.float64[:, :], norm_type),
         cache=True,
         parallel=True
         )
def calculate_core_dist(X_cluster, norm):
    d = X_cluster.shape[1]
    n = X_cluster.shape[0]
    core_dists = zeros(n, dtype=float64)
    for j in nb.prange(n):
        dist = norm(X_cluster, X_cluster[j])
        dist[j] = inf
        dist = dist ** -(d + 0.0)
        core_dists[j] = sum(dist) / (n - 1)
        core_dists[j] = core_dists[j] ** (-1 / d)

    return core_dists
