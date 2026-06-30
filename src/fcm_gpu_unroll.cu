#include "fcm_gpu.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#define Dim 41

#define CUDA_CHECK(call)                                            \
    do                                                              \
    {                                                               \
        cudaError_t err__ = (call);                                 \
        if (err__ != cudaSuccess)                                   \
        {                                                           \
            fprintf(stderr, "Errore CUDA in %s:%d -> %s\n",         \
                    __FILE__, __LINE__, cudaGetErrorString(err__)); \
            std::exit(1);                                           \
        }                                                           \
    } while (0)

// ---------------------------------------------------------------------------
// Kernel 1: calcolo delle distanze al quadrato punto-centroide.
// Un thread per punto i; ciascun thread scorre i K centroidi.
// Pattern di accesso regolare (coalesced sui dati del punto), nessun branch
// dipendente dai dati -> buon utilizzo dei warp.
// ---------------------------------------------------------------------------
__global__ void kernel_compute_distances(const float *__restrict__ data,
                                         const float *__restrict__ C,
                                         int N, int D, int K,
                                         float *__restrict__ dist)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N)
        return;

    const float *x = data + (size_t)i * D;
    float *di = dist + (size_t)i * K;

    for (int j = 0; j < K; ++j)
    {
        const float *c = C + (size_t)j * D;
        float s = 0.0f;
#pragma unroll
        for (int d = 0; d < Dim; ++d)
        {
            float diff = x[d] - c[d];
            s += diff * diff;
        }
        di[j] = (s < 1e-10f) ? 1e-10f : s;
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: aggiornamento delle appartenenze a partire dalle distanze gia'
// calcolate. Un thread per punto i.
//   u_ij = 1 / sum_k ( d_ij / d_ik )^(2/(m-1))
// ---------------------------------------------------------------------------
__global__ void kernel_update_memberships(const float *__restrict__ dist,
                                          int N, int K, float m,
                                          float *__restrict__ U)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N)
        return;

    float exponent = 2.0f / (m - 1.0f);
    const float *di = dist + (size_t)i * K;
    float *ui = U + (size_t)i * K;

    for (int j = 0; j < K; ++j)
    {
        float sum = 0.0f;
        for (int kk = 0; kk < K; ++kk)
            sum += powf(di[j] / di[kk], exponent);
        ui[j] = 1.0f / sum;
    }
}

// ---------------------------------------------------------------------------
// Kernel 3a: accumulo (numeratore, denominatore) per il calcolo dei centroidi.
//   num_j += u_ij^m * x_i      (per ogni dimensione d)
//   den_j += u_ij^m
// Un thread per punto i, con atomicAdd su num/den condivisi tra tutti i thread.
//
// NOTA DIDATTICA: questa e' la versione "base", corretta e semplice da
// seguire, ma con possibile contesa sugli atomicAdd se K*D e' grande.
// Una possibile ottimizzazione (lasciata come estensione) e' accumulare
// prima in shared memory per blocco e fare un solo atomicAdd per blocco
// invece che per thread.
// ---------------------------------------------------------------------------
__global__ void kernel_accumulate_centroids(const float *__restrict__ data,
                                            const float *__restrict__ U,
                                            int N, int D, int K, float m,
                                            float *__restrict__ num,
                                            float *__restrict__ den)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N)
        return;

    const float *x = data + (size_t)i * D;
    const float *u = U + (size_t)i * K;

    for (int j = 0; j < K; ++j)
    {
        float w = powf(u[j], m);
        atomicAdd(&den[j], w);
        float *nj = num + (size_t)j * D;
#pragma unroll
        for (int d = 0; d < Dim; ++d)
            atomicAdd(&nj[d], w * x[d]);
    }
}

// Kernel 3b: finalizza i centroidi dividendo numeratore/denominatore.
// Un thread per cluster j.
__global__ void kernel_finalize_centroids(const float *__restrict__ num,
                                          const float *__restrict__ den,
                                          int K, int D,
                                          float *__restrict__ C)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= K)
        return;

    float denom = (den[j] < 1e-12f) ? 1e-12f : den[j];
    float *cj = C + (size_t)j * D;
    const float *nj = num + (size_t)j * D;
#pragma unroll
    for (int d = 0; d < Dim; ++d)
        cj[d] = nj[d] / denom;
}

// ---------------------------------------------------------------------------
// Kernel 4: calcolo di max(|U_new - U_prev|) per il criterio di convergenza.
// Si usa atomicMax su unsigned int interpretando il bit-pattern del float
// come intero: per valori >= 0 (come un valore assoluto) l'ordinamento dei
// bit pattern IEEE-754 coincide con l'ordinamento dei valori float, quindi
// il trucco e' corretto e ci evita di scaricare l'intera matrice U sull'host
// ad ogni iterazione (servirebbe solo per il check di convergenza).
// ---------------------------------------------------------------------------
__global__ void kernel_max_diff(const float *__restrict__ U_new,
                                const float *__restrict__ U_prev,
                                int size, unsigned int *max_bits)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
        return;

    float diff = fabsf(U_new[idx] - U_prev[idx]);
    atomicMax(max_bits, __float_as_uint(diff));
}

// ---------------------------------------------------------------------------
// Driver host: alloca memoria device, copia i dati, esegue il ciclo di
// iterazioni Fuzzy C-Means e riporta i risultati sull'host.
// ---------------------------------------------------------------------------
double fcm_gpu_run(const float *data, int N, int D, int K, float m,
                   int max_iter, float tol,
                   const float *U_init,
                   float *U_out, float *C_out, int *iters_out, int nThreads)
{
    float *d_data = nullptr, *d_U = nullptr, *d_U_prev = nullptr, *d_C = nullptr;
    float *d_dist = nullptr, *d_num = nullptr, *d_den = nullptr;
    unsigned int *d_max_bits = nullptr;

    size_t data_bytes = (size_t)N * D * sizeof(float);
    size_t U_bytes = (size_t)N * K * sizeof(float);
    size_t C_bytes = (size_t)K * D * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_data, data_bytes));
    CUDA_CHECK(cudaMalloc(&d_U, U_bytes));
    CUDA_CHECK(cudaMalloc(&d_U_prev, U_bytes));
    CUDA_CHECK(cudaMalloc(&d_C, C_bytes));
    CUDA_CHECK(cudaMalloc(&d_dist, U_bytes)); // N x K, come U
    CUDA_CHECK(cudaMalloc(&d_num, C_bytes));  // K x D, come C
    CUDA_CHECK(cudaMalloc(&d_den, (size_t)K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_max_bits, sizeof(unsigned int)));

    CUDA_CHECK(cudaMemcpy(d_data, data, data_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_U, U_init, U_bytes, cudaMemcpyHostToDevice));

    const int threads = (nThreads > 0) ? nThreads : 256;
    const int blocksN = (N + threads - 1) / threads;
    const int blocksK = (K + threads - 1) / threads;
    const int blocksNK = ((N * K) + threads - 1) / threads;
    printf("GPU kernel launch configuration: %d threads/block, %d blocks for N, %d blocks for K, %d blocks for N*K\n",
           threads, blocksN, blocksK, blocksNK);
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    int it = 0;
    for (; it < max_iter; ++it)
    {
        // --- update centroidi ---
        CUDA_CHECK(cudaMemset(d_num, 0, C_bytes));
        CUDA_CHECK(cudaMemset(d_den, 0, (size_t)K * sizeof(float)));
        kernel_accumulate_centroids<<<blocksN, threads>>>(d_data, d_U, N, D, K, m, d_num, d_den);
        kernel_finalize_centroids<<<blocksK, threads>>>(d_num, d_den, K, D, d_C);

        // --- salva U corrente per il check di convergenza ---
        CUDA_CHECK(cudaMemcpy(d_U_prev, d_U, U_bytes, cudaMemcpyDeviceToDevice));

        // --- update appartenenze ---
        kernel_compute_distances<<<blocksN, threads>>>(d_data, d_C, N, D, K, d_dist);
        kernel_update_memberships<<<blocksN, threads>>>(d_dist, N, K, m, d_U);

        // --- check convergenza ---
        unsigned int zero_bits = 0;
        CUDA_CHECK(cudaMemcpy(d_max_bits, &zero_bits, sizeof(unsigned int), cudaMemcpyHostToDevice));
        kernel_max_diff<<<blocksNK, threads>>>(d_U, d_U_prev, N * K, d_max_bits);

        unsigned int max_bits_host = 0;
        CUDA_CHECK(cudaMemcpy(&max_bits_host, d_max_bits, sizeof(unsigned int), cudaMemcpyDeviceToHost));
        float max_diff;
        std::memcpy(&max_diff, &max_bits_host, sizeof(float));

        if (max_diff < tol)
        {
            ++it;
            break;
        }
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    CUDA_CHECK(cudaMemcpy(U_out, d_U, U_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(C_out, d_C, C_bytes, cudaMemcpyDeviceToHost));
    if (iters_out)
        *iters_out = it;

    cudaFree(d_data);
    cudaFree(d_U);
    cudaFree(d_U_prev);
    cudaFree(d_C);
    cudaFree(d_dist);
    cudaFree(d_num);
    cudaFree(d_den);
    cudaFree(d_max_bits);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return (double)elapsed_ms;
}
