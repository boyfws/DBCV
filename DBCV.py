from src.dbcv import dbcv

import numpy as np

from src.norms import euclidean_squared

from src.exceptions import *


def DBCV(X,
         labels,
         noise_id=-1,
         norm="euclidean_squared"):
    """Computes DBCV

    This function does not compute or store the distance matrix in memory
    using a lazy-computation approach and is optimized
    using parallel computation and numba

    Parameters
    ----------
    X :
        nd.array convertible to float
        shape (N, D)
        Sample embeddings

    labels:
        nd.array convertible to int if strict == False
        shape (N,)
        Cluster IDs assigned for each sample in X

    noise_id: default = -1
        int
        or
        iterable of int-s
        id or id-s of noise clusters

    norm: default = "euclidean_squared"
        str
        One of available norms (will be added soon)
    """
    if not isinstance(X, np.ndarray) or not isinstance(labels, np.ndarray):
        raise WrongInputDataError("Input data must be in np.ndarray format")

    if X.shape[0] != labels.shape[0]:
        raise WrongInputDataError("X and labels must have the same length")

    if X.ndim != 2 or labels.ndim != 1:
        raise WrongInputDataError(
            f"X and labels must have 2 and 1 numbers of dimensions respectively but were received {X.ndim} and {labels.ndim}")

    if not hasattr(noise_id, '__iter__') and not isinstance(noise_id, int):
        raise WrongInputDataError("noise_id must be int or iterable")
    elif hasattr(noise_id, '__iter__'):
        noise_id = list(noise_id)
    else:
        noise_id = [noise_id]

    n = X.shape[0]
    un_labels, counts = np.unique(labels, return_counts=True)
    mask_for_un_labels = ~((counts == 1) + np.isin(un_labels, noise_id))

    if np.sum(mask_for_un_labels) in [0, 1]:
        return 0

    mask_for_labels_and_X = np.isin(labels, un_labels[mask_for_un_labels])

    match norm:
        case "euclidean_squared":
            norm_for_func = euclidean_squared
        case _:
            raise WrongNormError("Norm is not available")

    return dbcv(X[mask_for_labels_and_X].astype(np.float64),
                labels[mask_for_labels_and_X].astype(np.int32),
                un_labels[mask_for_un_labels].astype(np.int32),
                counts[mask_for_un_labels].astype(np.int32),
                norm_for_func
                ) / n
