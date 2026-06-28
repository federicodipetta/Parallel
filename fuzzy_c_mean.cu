#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                             \
    do                                                                               \
    {                                                                                \
        cudaError_t err = call;                                                      \
        if (err != cudaSuccess)                                                      \
        {                                                                            \
            std::cerr << "Errore CUDA in " << __FILE__ << " alla linea " << __LINE__ \
                      << ": " << cudaGetErrorString(err) << std::endl;               \
            exit(EXIT_FAILURE);                                                      \
        }                                                                            \
    } while (0)

#define BLOCK_SIZE_RED 256

// 1. KERNEL: Aggiorna Matrice U
__global__ void kernel_aggiorna_U(const float *__restrict__ X, const float *__restrict__ centroidi,
                                  float *U, int n_punti, int n_cluster, int n_features, float m)
{
    int idx_punto = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_cluster = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx_punto < n_punti && idx_cluster < n_cluster)
    {
        float p = 2.0f / (m - 1.0f);
        float dist_corrente = 0.0f;
        for (int f = 0; f < n_features; f++)
        {
            float diff = X[idx_punto * n_features + f] - centroidi[idx_cluster * n_features + f];
            dist_corrente += diff * diff;
        }
        dist_corrente = sqrtf(dist_corrente);
        if (dist_corrente < 1e-8f)
            dist_corrente = 1e-8f;

        float somma = 0.0f;
        for (int c = 0; c < n_cluster; c++)
        {
            float dist_c = 0.0f;
            for (int f = 0; f < n_features; f++)
            {
                float diff = X[idx_punto * n_features + f] - centroidi[c * n_features + f];
                dist_c += diff * diff;
            }
            dist_c = sqrtf(dist_c);
            if (dist_c < 1e-8f)
                dist_c = 1e-8f;

            somma += powf((dist_corrente / dist_c), p);
        }
        U[idx_punto * n_cluster + idx_cluster] = 1.0f / somma;
    }
}

// 2. KERNEL: Calcola Centroidi
__global__ void kernel_calcola_centroidi(const float *__restrict__ X, const float *__restrict__ U,
                                         float *centroidi, int n_punti, int n_cluster, int n_features, float m)
{
    int idx_cluster = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_feature = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx_cluster < n_cluster && idx_feature < n_features)
    {
        float numeratore = 0.0f;
        float denominatore = 0.0f;
        for (int i = 0; i < n_punti; i++)
        {
            float u_m = powf(U[i * n_cluster + idx_cluster], m);
            numeratore += u_m * X[i * n_features + idx_feature];
            denominatore += u_m;
        }
        if (denominatore < 1e-8f)
            denominatore = 1e-8f;
        centroidi[idx_cluster * n_features + idx_feature] = numeratore / denominatore;
    }
}

// 3. KERNEL: Riduzione Parallela per Errore di Convergenza
__global__ void kernel_differenza_U_gpu(const float *U, const float *U_vecchia, float *d_errore_parziale, int size)
{
    __shared__ float sdata[BLOCK_SIZE_RED];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float diff = 0.0f;
    if (i < size)
    {
        float d = U[i] - U_vecchia[i];
        diff = d * d;
    }
    sdata[tid] = diff;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        d_errore_parziale[blockIdx.x] = sdata[0];
    }
}

int main()
{
    // --- Configurazione Parametri (Aumentati per Stress Test) ---
    const int N_PUNTI = 100000; // 100k punti
    const int N_CLUSTER = 5;    // 5 cluster
    const int N_FEATURES = 3;   // 3 dimensioni
    const float M_FUZZY = 2.0f;
    const int MAX_ITER = 50;
    const float TOLLERANZA = 1e-4f;

    size_t size_X = N_PUNTI * N_FEATURES * sizeof(float);
    size_t size_U = N_PUNTI * N_CLUSTER * sizeof(float);
    size_t size_C = N_CLUSTER * N_FEATURES * sizeof(float);

    // --- Dichiarazione Eventi CUDA per i Tempi ---
    cudaEvent_t start_totale, stop_totale;
    cudaEvent_t start_ciclo, stop_ciclo;
    CUDA_CHECK(cudaEventCreate(&start_totale));
    CUDA_CHECK(cudaEventCreate(&stop_totale));
    CUDA_CHECK(cudaEventCreate(&start_ciclo));
    CUDA_CHECK(cudaEventCreate(&stop_ciclo));

    // Avvia cronometro totale (Inizio Setup CPU)
    CUDA_CHECK(cudaEventRecord(start_totale));

    // Allocazione CPU
    float *h_X = (float *)malloc(size_X);
    float *h_U = (float *)malloc(size_U);
    float *h_centroidi = (float *)malloc(size_C);

    srand(42);
    for (int i = 0; i < N_PUNTI * N_FEATURES; i++)
        h_X[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < N_PUNTI; i++)
    {
        float somma = 0.0f;
        for (int j = 0; j < N_CLUSTER; j++)
        {
            h_U[i * N_CLUSTER + j] = (float)rand() / RAND_MAX;
            somma += h_U[i * N_CLUSTER + j];
        }
        for (int j = 0; j < N_CLUSTER; j++)
            h_U[i * N_CLUSTER + j] /= somma;
    }

    // Allocazione GPU
    float *d_X, *d_U, *d_U_vecchia, *d_centroidi, *d_errore_parziale;
    CUDA_CHECK(cudaMalloc((void **)&d_X, size_X));
    CUDA_CHECK(cudaMalloc((void **)&d_U, size_U));
    CUDA_CHECK(cudaMalloc((void **)&d_U_vecchia, size_U));
    CUDA_CHECK(cudaMalloc((void **)&d_centroidi, size_C));

    int num_blocks_reduction = (N_PUNTI * N_CLUSTER + BLOCK_SIZE_RED - 1) / BLOCK_SIZE_RED;
    float *h_errore_parziale = (float *)malloc(num_blocks_reduction * sizeof(float));
    CUDA_CHECK(cudaMalloc((void **)&d_errore_parziale, num_blocks_reduction * sizeof(float)));

    // Copia iniziale dei dati su GPU
    CUDA_CHECK(cudaMemcpy(d_X, h_X, size_X, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_U, h_U, size_U, cudaMemcpyHostToDevice));

    // Configurazione Griglie
    dim3 block_U(16, 16);
    dim3 grid_U((N_PUNTI + block_U.x - 1) / block_U.x, (N_CLUSTER + block_U.y - 1) / block_U.y);
    dim3 block_C(16, 16);
    dim3 grid_C((N_CLUSTER + block_C.x - 1) / block_C.x, (N_FEATURES + block_C.y - 1) / block_C.y);

    std::cout << "Data setup completato. Avvio ciclo FCM..." << std::endl;

    // ----------------------------------------------------
    // AVVIO CRONOMETRO DEL SOLO CICLO ALGORITMICO
    // ----------------------------------------------------
    CUDA_CHECK(cudaEventRecord(start_ciclo));

    int iterazione = 0;
    for (iterazione = 0; iterazione < MAX_ITER; iterazione++)
    {
        // Salvataggio U su GPU (Velocissimo, senza passare da PCIe)
        CUDA_CHECK(cudaMemcpy(d_U_vecchia, d_U, size_U, cudaMemcpyDeviceToDevice));

        // 1. Calcolo Centroidi
        kernel_calcola_centroidi<<<grid_C, block_C>>>(d_X, d_U, d_centroidi, N_PUNTI, N_CLUSTER, N_FEATURES, M_FUZZY);

        // 2. Aggiornamento Matrice U
        kernel_aggiorna_U<<<grid_U, block_U>>>(d_X, d_centroidi, d_U, N_PUNTI, N_CLUSTER, N_FEATURES, M_FUZZY);

        // 3. Riduzione Errore direttamente in GPU
        kernel_differenza_U_gpu<<<num_blocks_reduction, BLOCK_SIZE_RED>>>(d_U, d_U_vecchia, d_errore_parziale, N_PUNTI * N_CLUSTER);

        // 4. Copia del solo array di riduzione ridotto per la verifica
        CUDA_CHECK(cudaMemcpy(h_errore_parziale, d_errore_parziale, num_blocks_reduction * sizeof(float), cudaMemcpyDeviceToHost));

        float errore_totale = 0.0f;
        for (int b = 0; b < num_blocks_reduction; b++)
            errore_totale += h_errore_parziale[b];
        errore_totale = sqrtf(errore_totale);

        if (errore_totale < TOLLERANZA)
        {
            std::cout << "Convergenza raggiunta all'iterazione: " << iterazione << " (Errore: " << errore_totale << ")" << std::endl;
            break;
        }
    }

    // Stop cronometro ciclo
    CUDA_CHECK(cudaEventRecord(stop_ciclo));
    // ----------------------------------------------------

    // Recupero dei centroidi finali per l'Host
    CUDA_CHECK(cudaMemcpy(h_centroidi, d_centroidi, size_C, cudaMemcpyDeviceToHost));

    // Stop cronometro totale
    CUDA_CHECK(cudaEventRecord(stop_totale));

    // Sincronizzazione obbligatoria prima di leggere i tempi
    CUDA_CHECK(cudaEventSynchronize(stop_totale));
    CUDA_CHECK(cudaEventSynchronize(stop_ciclo));

    // Calcolo effettivo dei millisecondi passati
    float tempo_ciclo_ms = 0;
    float tempo_totale_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&tempo_ciclo_ms, start_ciclo, stop_ciclo));
    CUDA_CHECK(cudaEventElapsedTime(&tempo_totale_ms, start_totale, stop_totale));

    // --- Report Finali ---
    std::cout << "\n================ BENCHMARK RISULTATI ================" << std::endl;
    std::cout << "Tempo di computazione puro del ciclo FCM: " << tempo_ciclo_ms << " ms" << std::endl;
    std::cout << "Tempo medio per singola iterazione:       " << (tempo_ciclo_ms / (iterazione + 1)) << " ms" << std::endl;
    std::cout << "Tempo totale programma (Incluso I/O):     " << tempo_totale_ms << " ms" << std::endl;
    std::cout << "=====================================================" << std::endl;

    // Pulizia Eventi
    cudaEventDestroy(start_totale);
    cudaEventDestroy(stop_totale);
    cudaEventDestroy(start_ciclo);
    cudaEventDestroy(stop_ciclo);

    // Liberazione Memoria
    free(h_X);
    free(h_U);
    free(h_centroidi);
    free(h_errore_parziale);
    cudaFree(d_X);
    cudaFree(d_U);
    cudaFree(d_U_vecchia);
    cudaFree(d_centroidi);
    cudaFree(d_errore_parziale);

    return 0;
}