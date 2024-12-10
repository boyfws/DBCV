import numba as nb

# Dtypes
from numpy import float64

# Array creation
from numpy import zeros

# Other
from numpy import maximum, max


@nb.njit(nb.float64(
    nb.int32[:],
    nb.float64[:],
    nb.int32[:],
    nb.int32[:],
    nb.float64[:]
    ),
    cache=True,
    parallel=True)
def find_dsbcs(columns, core_dists, row, col, weights):

    columns_set = set(columns)
    val = zeros(row.size, dtype=float64)

    for j in nb.prange(row.size):
        if row[j] in columns_set and col[j] in columns_set:
            val[j] = maximum(weights[j],
                maximum(
                    core_dists[col[j]], core_dists[row[j]]
                )
            )
    return max(val)
