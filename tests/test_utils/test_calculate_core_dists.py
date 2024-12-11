"""
Нам нужно проверить что декорирование не повлияет на значение функции
"""


import pathlib
import sys
import pytest

import numpy as np

func_path = pathlib.Path().cwd().parent.parent / "src" / "dbcv" / "utils"
norm_path = pathlib.Path().cwd().parent.parent / "src"

sys.path.append(str(func_path.resolve()))
sys.path.append(str(norm_path.resolve()))

from calculate_core_dist import calculate_core_dist
from norms import euclidean_squared


@pytest.fixture(scope="function")
def ord_func():
    def calculate_core_dist_real(X_cluster, norm):
        d = X_cluster.shape[1]
        n = X_cluster.shape[0]
        core_dists = np.zeros(n, dtype=np.float64)
        for j in range(n):
            dist = norm(X_cluster, X_cluster[j])
            dist[j] = np.inf
            dist = dist ** -(d + 0.0)
            core_dists[j] = sum(dist) / (n - 1)

        return core_dists ** (-1 / d)
    return calculate_core_dist_real


@pytest.fixture(scope="function")
def euclidean_sq_norm():
    return euclidean_squared


## Мы провреяем, что у декорированной функции результаты прмерно те же
@pytest.mark.parametrize(
    "X_cluster",
    [
        np.array([[0.0], [1.0]], dtype=np.float64),   # Две точки
        np.array([[0.0], [1.0], [2.0]], dtype=np.float64),   # Три точки
        np.random.rand(10 ** 3, 2).astype(np.float64),   # Случайные точки
    ],
)
def test_similarity_sq_eq_norm(ord_func, euclidean_sq_norm, X_cluster):
    ord_res = ord_func(X_cluster, euclidean_sq_norm)
    dec_res = calculate_core_dist(X_cluster, euclidean_sq_norm)
    assert pytest.approx(ord_res) ==  dec_res

