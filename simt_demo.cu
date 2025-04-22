#include <stdio.h>

// CUDA Kernel (SIMT-Instruction)
__global__ void multiplyByTwo(int *output) {
    // Thread unique index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Same instruction for every thread
    output[tid] = tid * 2;
}

int main() {
    const int N = 256;  // Total amount of threads
    int *d_output;  // Pointer to GPU Memory

    // 1. Allocate GPU Memory
    cudaMalloc(&d_output, N * sizeof(int));

    // 2. Run kernel (8 block with 32 threads)
    multiplyByTwo<<<8, 32>>>(d_output);

    // 3. Copy results to CPU
    int host_output[N];
    cudaMemcpy(host_output, d_output, N * sizeof(int), cudaMemcpyDeviceToHost);

    // 4. Print results
    for (int i = 0; i < N; ++i) {
        printf("Thread %3d: %d * 2 = %d\n", i, i, host_output[i]);
    }

    // 5. Free memory
    cudaFree(d_output);

    return 0;
}