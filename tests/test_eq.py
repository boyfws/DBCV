"""
Мы проверяем что значения нашей метрики равны
https://github.com/FelSiq/DBCV
Поправка в 7% уместна, так как метрика включает в себя композицию разрывных функций
(MST - очень сильно зависит от точности типа данных для плотной матрицы (у нас именно такая)
после к результатам MST применяется еще одна разрывная функция
"""
from pathlib import Path
import sys

DBCV_path = Path().cwd().parent

sys.path.append(str(DBCV_path))

from DBCV import DBCV 
import sklearn.datasets
import numpy as np


def test_first_sample():
    X, y = sklearn.datasets.make_moons(n_samples=300, noise=0.05, random_state=1782)

    score = DBCV(X, y)
    expected = 0.8545358723390613
    assert expected * 0.93 <= score <= expected * 1.07


def test_second_sample():
    X, y = sklearn.datasets.make_moons(n_samples=500, noise=0.05, random_state=1782)

    noise_id = -1 # Дефолтное значение свовпадает

    rng = np.random.RandomState(1082)
    X_noise = rng.uniform(*np.quantile(X, (0, 1), axis=0), size=(100, 2))
    y_noise = 100 * [noise_id]

    X, y = np.vstack((X, X_noise)), np.hstack((y, y_noise))

    score = DBCV(X, y, noise_id=noise_id)
    expected = 0.7545431212217051
    assert expected * 0.93 <= score <= expected * 1.07
