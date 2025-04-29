import numpy as np
from numba import cuda 

# datasize
N = 1000

# data initialization
A = np.full(N, 2.0, dtype=np.float32) # [2.0, 2.0, ...]
B = np.full(N, 3.0, dtype=np.float32) # [3.0, 3.0, ...]
C = np.zeros(N, dtype=np.float32)

# CPU function: elementary multipy
def vector_mul_cpu(a, b, c):
    for i in range(len(a)):
        c[i] = a[i] * b[i]

# GPU function: add 10 to every element
@cuda.jit
def vector_add_ten_gpu(c):
    idx = cuda.grid(1)
    if idx < len(c):
        c[idx] += 10.0

# === CPU part ===
vector_mul_cpu(A, B, C) # C[i] = A[i] * B[i]

print(f"CPU result (first 3): {C[:3]} ...")

# === GPU part ===
# copy data to device
d_C = cuda.to_device(C)

# runtime configuration
threads_per_block = 256
blocks_per_grid = (N + threads_per_block - 1) // threads_per_block

vector_add_ten_gpu[blocks_per_grid, threads_per_block](d_C) # C[i] += 10
cuda.synchronize() # wait for GPU finish

# copy data back
d_C.copy_to_host(C)

print(f"Final result (first 3): {C[:3]} ...")