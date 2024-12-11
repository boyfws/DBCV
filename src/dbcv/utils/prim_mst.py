# Types
from numpy import inf, bool_, float64, int32

# Creating array
from numpy import zeros, full

# Other
from numpy import flatnonzero, maximum, arange

# Numba types
import numba as nb
import heapq

from ...config import norm_type


@nb.njit(nb.types.Tuple((
        nb.int32[:],
        nb.int32[:],
        nb.float64[:]))(nb.float64[:, :],
                        nb.float64[:],
                        norm_type),
         cache=True)
def prim_mst(observations, core_dist, norm):
    n = observations.shape[0]
    in_tree = zeros(n, dtype=bool_)
    min_weight = full(n, inf, float64)
    previous = full(n, -1, dtype=int32)
    start_vertex = 0
    min_weight[start_vertex] = 0
    heap = [(float64(0), start_vertex)]

    while heap:
        current_min_weight, u = heapq.heappop(heap)

        if in_tree[u]:
            continue

        in_tree[u] = True

        not_in_tree = flatnonzero(~in_tree)

        distances = norm(observations[not_in_tree], observations[u])
        distances = maximum(distances, core_dist[not_in_tree])
        distances = maximum(distances, core_dist[u])

        # Нужно для совместимости с функцией из scipy
        distances[distances < 1e-8] = 0.0

        mask = distances < min_weight[not_in_tree]
        min_weight[not_in_tree[mask]] = distances[mask]
        previous[not_in_tree[mask]] = u

        for v in not_in_tree[mask]:
            heapq.heappush(heap, (min_weight[v], v))

    return arange(1, n, dtype=int32), previous[1:], min_weight[1:]