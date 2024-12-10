from numba import float64
from numba.types import FunctionType


signature_for_norms = float64[:](float64[:, :], float64[:])

norm_type = FunctionType(signature_for_norms)
