#include "fcm_cpu.h"
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>

// Calcola i nuovi centroidi a partire dalle appartenenze correnti U:
//   C_j = ( sum_i u_ij^m * x_i ) / ( sum_i u_ij^m )
// Si usa un accumulatore in double per ridurre l'errore numerico su N grandi.
static void update_centroids(const float* data, const float* U,
                              int N, int D, int K, float m,
                              float* C) {
    std::vector<double> num((size_t)K * D, 0.0);
    std::vector<double> den((size_t)K, 0.0);

    for (int i = 0; i < N; ++i) {
        const float* x = data + (size_t)i * D;
        const float* u = U + (size_t)i * K;
        for (int j = 0; j < K; ++j) {
            double w = std::pow((double)u[j], (double)m);
            den[j] += w;
            double* nj = &num[(size_t)j * D];
            for (int d = 0; d < D; ++d)
                nj[d] += w * (double)x[d];
        }
    }

    for (int j = 0; j < K; ++j) {
        double denom = (den[j] < 1e-12) ? 1e-12 : den[j];
        for (int d = 0; d < D; ++d)
            C[(size_t)j * D + d] = (float)(num[(size_t)j * D + d] / denom);
    }
}

static inline double squared_dist(const float* x, const float* c, int D) {
    double s = 0.0;
    for (int d = 0; d < D; ++d) {
        double diff = (double)x[d] - (double)c[d];
        s += diff * diff;
    }
    return s;
}

// Aggiorna le appartenenze a partire dai centroidi correnti:
//   u_ij = 1 / sum_k ( d_ij / d_ik )^(2/(m-1))
// Le distanze sono clampate a un epsilon per evitare divisioni per zero.
// NB: stesso clamp usato nel kernel GPU, per rendere i due risultati
// confrontabili (vedi fcm_gpu.cu).
static void update_memberships(const float* data, const float* C,
                                int N, int D, int K, float m,
                                float* U) {
    const double eps = 1e-10;
    const double exponent = 2.0 / ((double)m - 1.0);
    std::vector<double> dist((size_t)K);

    for (int i = 0; i < N; ++i) {
        const float* x = data + (size_t)i * D;

        for (int j = 0; j < K; ++j) {
            double d2 = squared_dist(x, C + (size_t)j * D, D);
            dist[j] = (d2 < eps) ? eps : d2;
        }

        float* u = U + (size_t)i * K;
        for (int j = 0; j < K; ++j) {
            double sum = 0.0;
            for (int kk = 0; kk < K; ++kk)
                sum += std::pow(dist[j] / dist[kk], exponent);
            u[j] = (float)(1.0 / sum);
        }
    }
}

double fcm_cpu_run(const float* data, int N, int D, int K, float m,
                    int max_iter, float tol,
                    const float* U_init,
                    float* U_out, float* C_out, int* iters_out) {
    std::vector<float> U(U_init, U_init + (size_t)N * K);
    std::vector<float> U_prev(U.size());
    std::vector<float> C((size_t)K * D);

    auto t0 = std::chrono::high_resolution_clock::now();

    int it = 0;
    for (; it < max_iter; ++it) {
        update_centroids(data, U.data(), N, D, K, m, C.data());

        U_prev = U;
        update_memberships(data, C.data(), N, D, K, m, U.data());

        float max_diff = 0.0f;
        for (size_t idx = 0; idx < U.size(); ++idx)
            max_diff = std::max(max_diff, std::fabs(U[idx] - U_prev[idx]));

        if (max_diff < tol) { ++it; break; }
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    std::copy(U.begin(), U.end(), U_out);
    std::copy(C.begin(), C.end(), C_out);
    if (iters_out) *iters_out = it;

    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}
