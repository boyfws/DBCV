from DBCV import DBCV
import subprocess
import timeit
import sklearn.datasets
from itertools import product
import numba as nb

print(f"Кол-во используемых потоков: {nb.get_num_threads()}")

nb.set_num_threads(4)

command = ["python", "-m", "pip", "install", "git+https://github.com/FelSiq/DBCV"]

try:
    subprocess.run(command, check=True)
    print("Установка завершена успешно.")
except subprocess.CalledProcessError as e:
    print(f"Ошибка при выполнении команды: {e}")

from dbcv import dbcv


# make_blobs
def make_blobs():
    print("Датасет make_blobs")
    n_samples_list = [10 ** 3, 10 ** 4]
    vec_size_list = [6, 8]
    centers_list = [2, 4, 8, 10]
    for n_samples, vec_size, centers in product(n_samples_list, vec_size_list, centers_list):
        X, y = sklearn.datasets.make_blobs(n_samples=n_samples,
                                           centers=centers,
                                           n_features=vec_size,
                                           random_state=0)
        ord_func = timeit.timeit(lambda: dbcv(X, y, n_processes=1), number=10) / 10
        my_func = timeit.timeit(lambda: DBCV(X, y), number=10) / 10 
        print(f"Обычная функция: Размер выборки ({n_samples}, {vec_size}), кол-во центров: {centers}, время {ord_func}")
        print(f"Моя функция: Размер выборки ({n_samples}, {vec_size}), кол-во центров: {centers}, время {my_func}")


make_blobs()





