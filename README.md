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