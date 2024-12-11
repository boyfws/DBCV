# Function for calculating Density-Based Clustering Validation metric  
This function is optimized with numba and multithreading (Not python library) and does not store the distance matrix in memory but computes it inplace using lazy computation approach
## A little bit about implementation
1) We use Prim's algorithm to find mst
2) 
   So far, only the square of Euclid's norm is available 
    
    However, it's easy to add the norm you need to this function: wrap the function to compute the norm in an njit decorator with 'signature_for_norms' from src.config like this: 

    ```python
        @numba.njit(signature_for_norms, cache=True)
        def your_norm(x, y):
            ...
    ```
3) You can control the number of threads in use via numba.set_num_threads()

## Speed

**Tests were run on the make_blobs dataset from sklearn.datasets**

On average, 15 times faster than realizations from https://github.com/FelSiq/DBCV

### Sample size == 10^4

| Vector length | Num. of clusters | Av. time of 10 runs for "old" DBCV sec. | Av. time of 10 runs for "new" DBCV sec. |
|---------------|------------------|-----------------------------------------|-----------------------------------------|
| 6             | 2                | 26.86                                   | 1.23                                    |
| 6             | 4                | 11.33                                   | 0.60                                    |
| 6             | 8                | 5.47                                    | 0.34                                    |
| 6             | 10               | 4.33                                    | 0.28                                    |
| 8             | 2                | 26.53                                   | 1.41                                    |
| 8             | 4                | 11.41                                   | 0.69                                    |
| 8             | 8                | 5.23                                    | 0.37                                    |
| 8             | 10               | 4.42                                    | 0.34                                    |


### Sample size == 10^3


| Vector length | Num. of clusters | Av. time of 10 runs for "old" DBCV sec. | Av. time of 10 runs for "new" DBCV sec. |
|---------------|------------------|-----------------------------------------|-----------------------------------------|
| 6             | 2                | 0.209                                   | 0.014                                   |
| 6             | 4                | 0.147                                   | 0.012                                   |
| 6             | 8                | 0.101                                   | 0.008                                   |
| 6             | 10               | 0.102                                   | 0.008                                   |
| 8             | 2                | 0.221                                   | 0.018                                   |
| 8             | 4                | 0.132                                   | 0.010                                   |
| 8             | 8                | 0.102                                   | 0.008                                   |
| 8             | 10               | 0.098                                    | 0.008                                   |

## Future plans

1) Currently the bottleneck of the function is the mst build, which is done in a single thread. It is planned to switch to Boruvka's algorithm, which can be executed in parallel 
2) There are also plans to add support for execution on gpu 

