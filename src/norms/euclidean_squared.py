import numba as nb
import numpy as np

from ..config import signature_for_norms


@nb.njit(signature_for_norms, cache=True)
def euclidean_squared(x, y):
    return np.sum((x - y) ** 2, axis=1)
