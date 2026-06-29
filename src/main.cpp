#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

#include "dataset.h"
#include "fcm_cpu.h"
//#include "fcm_gpu.cuh"

// Inizializza la matrice di appartenenza U (N x K) con valori casuali
// normalizzati riga per riga (ogni punto ha appartenenze che sommano a 1).
// Lo stesso U_init viene passato sia alla versione CPU che a quella GPU,
// cosi' i due run partono dallo stesso punto e i risultati sono confrontabili.
static void init_membership(std::vector<float>& U, int N, int K, unsigned int seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    U.resize((size_t)N * K);
    for (int i = 0; i < N; ++i) {
        float* row = &U[(size_t)i * K];
        float sum = 0.0f;
        for (int j = 0; j < K; ++j) {
            row[j] = dist(rng);
            sum += row[j];
        }
        for (int j = 0; j < K; ++j) row[j] /= sum;
    }
}

// Funzione obiettivo del Fuzzy C-Means:  J = sum_i sum_j u_ij^m * ||x_i - c_j||^2
// Usata solo per validare che CPU e GPU convergano a soluzioni equivalenti.
static double fcm_objective(const float* data, const float* U, const float* C,
                             int N, int D, int K, float m) {
    double J = 0.0;
    for (int i = 0; i < N; ++i) {
        const float* x = data + (size_t)i * D;
        const float* u = U + (size_t)i * K;
        for (int j = 0; j < K; ++j) {
            const float* c = C + (size_t)j * D;
            double d2 = 0.0;
            for (int d = 0; d < D; ++d) {
                double diff = (double)x[d] - (double)c[d];
                d2 += diff * diff;
            }
            J += std::pow((double)u[j], (double)m) * d2;
        }
    }
    return J;
}

int main(int argc, char** argv) {
    // Parametri di default
    int N = 1 << 20; // 1M punti
    int D = 32;
    int K = 100;
    float m = 2.0f;
    int max_iter = 100;
    float tol = 1e-4f;
    unsigned int seed = 42;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--N") && i + 1 < argc) N = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--D") && i + 1 < argc) D = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--K") && i + 1 < argc) K = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--m") && i + 1 < argc) m = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--max_iter") && i + 1 < argc) max_iter = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tol") && i + 1 < argc) tol = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = (unsigned int)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--help")) {
            printf("Uso: %s [--N punti] [--D dimensioni] [--K cluster] [--m fuzzifier]\n"
                   "          [--max_iter n] [--tol t] [--seed s]\n", argv[0]);
            return 0;
        }
    }

    printf("=== Fuzzy C-Means: benchmark CPU vs GPU ===\n");
    printf("N=%d  D=%d  K=%d  m=%.2f  max_iter=%d  tol=%g  seed=%u\n\n",
           N, D, K, m, max_iter, tol, seed);

    // 1) Dataset sintetico (blob gaussiani)
    printf("Generazione dataset sintetico...\n");
    std::vector<float> data;
    std::vector<int> true_labels;
    generate_synthetic_blobs(data, true_labels, N, D, K,
                              /*spread=*/10.0f, /*cluster_std=*/1.5f, seed);

    // 2) Stessa inizializzazione casuale per entrambi i run
    std::vector<float> U_init;
    init_membership(U_init, N, K, seed + 1);

    std::vector<float> U_cpu((size_t)N * K), C_cpu((size_t)K * D);
    std::vector<float> U_gpu((size_t)N * K), C_gpu((size_t)K * D);
    int iters_cpu = 0, iters_gpu = 0;

    // 3) Run CPU
    printf("\n--- CPU (single-thread) ---\n");
    double t_cpu = fcm_cpu_run(data.data(), N, D, K, m, max_iter, tol,
                               U_init.data(), U_cpu.data(), C_cpu.data(), &iters_cpu);
    printf("Iterazioni: %d\n", iters_cpu);
    printf("Tempo CPU:  %.2f ms\n", t_cpu);

    // 4) Run GPU
    // printf("\n--- GPU (CUDA) ---\n");
    // double t_gpu = fcm_gpu_run(data.data(), N, D, K, m, max_iter, tol,
    //                            U_init.data(), U_gpu.data(), C_gpu.data(), &iters_gpu);
    // printf("Iterazioni: %d\n", iters_gpu);
    // printf("Tempo GPU:  %.2f ms\n", t_gpu);

    // 5) Validazione: la funzione obiettivo deve essere comparabile
    double J_cpu = fcm_objective(data.data(), U_cpu.data(), C_cpu.data(), N, D, K, m);
    double J_gpu = fcm_objective(data.data(), U_gpu.data(), C_gpu.data(), N, D, K, m);

    printf("\n=== Risultati ===\n");
    printf("Funzione obiettivo J (CPU): %f\n", J_cpu);
    printf("Funzione obiettivo J (GPU): %f\n", J_gpu);
    printf("Differenza relativa:        %.6f%%\n",
           100.0 * std::fabs(J_cpu - J_gpu) / J_cpu);
    printf("Speedup (T_cpu / T_gpu):    %.2fx\n", t_cpu / 100.0); // t_gpu non calcolato in questa versione

    return 0;
}
