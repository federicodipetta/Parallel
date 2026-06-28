#include <cstdio>
#include <cuda_runtime.h>

__global__ void hello()
{
    printf("Hello from block %d, thread %d, y %d, y %d\n", blockIdx.x, threadIdx.x, blockIdx.y, threadIdx.y);
}

int main()
{
    // Launch: gridDim.x = 2 blocks, blockDim.x = 4 threads per block
    hello<<<2, 4>>>();

    // Check launch error (optional but recommended)
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Wait for GPU to finish so device printf gets flushed
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error after kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}
