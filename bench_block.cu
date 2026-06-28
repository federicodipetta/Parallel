#include <cstdio>
#include <cuda_runtime.h>
#define CUDA_CHECK(x)                                                                 \
    do                                                                                \
    {                                                                                 \
        cudaError_t e = x;                                                            \
        if (e != cudaSuccess)                                                         \
        {                                                                             \
            printf("Error %s:%d -> %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
            exit(1);                                                                  \
        }                                                                             \
    } while (0)

__global__ void addGlobal(const int *A, const int *B, int *C, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

__global__ void addShared(const int *A, const int *B, int *C, int N)
{
    __shared__ int sA[512];
    __shared__ int sB[512];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    if (blockDim.y >= 2)
    {
        printf("Error  -> blockDim.y is %d, should be 1\n", blockDim.y);
    }
    // printf("Thread %d in block %d: i=%d, dim=%d\n", tid, blockIdx.x, i, blockDim.x);
    if (i < N)
    {
        sA[tid] = A[i];
        sB[tid] = B[i];
    }
    __syncthreads();
    if (i < N)
        C[i] = sA[tid] + sB[tid];
}
int main()
{
    const int N = 1 << 24; // ~16M elements
    size_t bytes = N * sizeof(int);
    int *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));

    int blocks[3] = {128, 256, 512};

    for (int b = 0; b < 3; b++)
    {
        int block = blocks[b];
        int grid = (N + block - 1) / block;
        cudaEvent_t s, e;
        cudaEventCreate(&s);
        cudaEventCreate(&e);

        // ---------- GLOBAL ----------
        cudaEventRecord(s);
        addGlobal<<<grid, block>>>(dA, dB, dC, N);
        cudaEventRecord(e);
        cudaEventSynchronize(e);
        float tGlobal;
        cudaEventElapsedTime(&tGlobal, s, e);

        // ---------- SHARED ----------
        cudaEventRecord(s);
        addShared<<<grid, block>>>(dA, dB, dC, N);
        cudaEventRecord(e);
        cudaEventSynchronize(e);
        float tShared;
        cudaEventElapsedTime(&tShared, s, e);

        printf("Block=%3d | Global=%.3f ms | Shared=%.3f ms\n", block, tGlobal, tShared);
    }
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}
