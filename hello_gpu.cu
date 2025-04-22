#include <stdio.h>

// Function that runs using GPU (CUDA Core)
__global__ void helloFromGPU() {
    printf("Hello, GPU!\n");
} 

int main() {
    // Run core at GPU (1 block, 1 thread)
    helloFromGPU<<<1, 1>>>();

    // Synchronize CPU and GPU (to wait when GPU proccess ends)
    cudaDeviceSynchronize();

    return 0;
}