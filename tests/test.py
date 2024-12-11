"""
Тест базового функционала
"""
import time
from pathlib import Path
import sys
import numpy as np
import pytest
import numba as nb
from itertools import product

DBCV_path = Path().cwd().parent

sys.path.append(str(DBCV_path))

from DBCV import DBCV

n_samples_list = [10 ** 2, 10 ** 3, 10 ** 4]
vec_size_list = [2, 4, 6, 8]
num_checks_list = [50, 100]
norms_list = ["euclidean_squared"]


# Проверяем, что при перезапуске функции результат не меняется
# на случай если есть race condition или другие эффекты влияющие на результат
@pytest.mark.parametrize('n_samples, vec_size, num_checks, norm',
                         product(
                             n_samples_list,
                             vec_size_list,
                             num_checks_list,
                             norms_list
                            )
                         )
def test_multiple_reruns(n_samples, vec_size, num_checks, norm):
    X = np.random.standard_normal((n_samples, vec_size))
    y = np.random.randint(low=0, high=10, size=n_samples)

    res = [None] * num_checks
    for i in range(num_checks):
        res[i] = DBCV(X, y, norm=norm)
        time.sleep(0.5)

    last = res[-1]
    assert all(pytest.approx(last) == el for el in res)



@pytest.mark.parametrize('n_samples, vec_size, norm',
                         product(
                             n_samples_list,
                             vec_size_list,
                             norms_list
                            )
                         )
def test_multiple_threads(n_samples, vec_size, norm):
    start = 1
    end = 7

    X = np.random.standard_normal((n_samples, vec_size))
    y = np.random.randint(low=0, high=10, size=n_samples)

    res = [i for i in range(start, end + 1)]

    for i in range(start, end + 1):
        nb.set_num_threads(i)
        res[i - start] = DBCV(X, y, norm=norm)
        time.sleep(0.5)

    last = res[-1]

    assert all(pytest.approx(last) == el for el in res)





