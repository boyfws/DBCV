"""
Мы проверяем что значения нашей метрики равны
https://github.com/FelSiq/DBCV
Поправка в 7% уместна, так как метрика включает в себя композицию разрывных функций
(MST - очень сильно зависит от точности типа данных для плотной матрицы (у нас именно такая)
после к результатам MST применяется еще одна разрывная функция
"""
from pathlib import Path
import sys
import pandas as pd

import pytest

DBCV_path = Path().cwd().parent

sys.path.append(str(DBCV_path))

from DBCV import DBCV 
import sklearn.datasets
import numpy as np


threshold = 0.07


def test_first_sample():
    X, y = sklearn.datasets.make_moons(n_samples=300, noise=0.05, random_state=1782)

    score = DBCV(X, y)
    expected = 0.8545358723390613
    assert expected * (1 - threshold) <= score <= expected * (1 + threshold)


def test_second_sample():
    X, y = sklearn.datasets.make_moons(n_samples=500, noise=0.05, random_state=1782)

    noise_id = -1 # Дефолтное значение свовпадает

    rng = np.random.RandomState(1082)
    X_noise = rng.uniform(*np.quantile(X, (0, 1), axis=0), size=(100, 2))
    y_noise = 100 * [noise_id]

    X, y = np.vstack((X, X_noise)), np.hstack((y, y_noise))

    score = DBCV(X, y, noise_id=noise_id)
    expected = 0.7545431212217051
    assert  expected * (1 - threshold) <= score <= expected * (1 + threshold)


# Сравниваем с MATLAB реализацией, так как именно она использует алгоритм Прима
@pytest.mark.parametrize('number, expected', [
    (1, 0.8576),
    (2, 0.8103),
    (3, 0.6319),
    (4, 0.8688)
])
def test_matlab_datasets(number, expected):
    df = pd.read_csv(f"matlab_datasets/dataset_{number}.txt", header=None, sep=" ")

    df_np = df.to_numpy()

    X = df_np[:, :2]
    y = df_np[:, 2]
    score = DBCV(X, y)
    assert  expected * (1 - threshold) <= score <= expected * (1 + threshold)
