#include "dataset.h"
#include <random>

void generate_synthetic_blobs(std::vector<float>& data,
                               std::vector<int>& true_labels,
                               int N, int D, int K,
                               float spread, float cluster_std,
                               unsigned int seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> center_dist(-spread, spread);
    std::normal_distribution<float> noise_dist(0.0f, cluster_std);
    std::uniform_int_distribution<int> cluster_pick(0, K - 1);

    // 1) Genera K centri casuali
    std::vector<float> centers((size_t)K * D);
    for (int k = 0; k < K; ++k)
        for (int d = 0; d < D; ++d)
            centers[(size_t)k * D + d] = center_dist(rng);

    // 2) Genera N punti: per ognuno scegli un blob di appartenenza
    //    e aggiungi rumore gaussiano al centro corrispondente
    data.resize((size_t)N * D);
    true_labels.resize(N);

    for (int i = 0; i < N; ++i) {
        int k = cluster_pick(rng);
        true_labels[i] = k;
        const float* c = &centers[(size_t)k * D];
        float* x = &data[(size_t)i * D];
        for (int d = 0; d < D; ++d) {
            x[d] = c[d] + noise_dist(rng);
        }
    }
}
