#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

#include "fcm_cpu.h"
#include "parser.cpp"

#ifdef ONLY_GPU
static const int only_gpu = 1;
#else
static const int only_gpu = 0;
#endif

#ifdef ONLY_CPU
double fcm_gpu_run(const float *data, int N, int D, int K, float m,
                   int max_iter, float tol,
                   const float *U_init,
                   float *U_out, float *C_out, int *iters_out, int nThreads = 0) { return 1.0; }
#else
#include "fcm_gpu.cuh"
#endif

const std::string USAGE = "USAGE: ./fcm [--N points] [--D dimensions] [--K clusters] [--m fuzzifier]\n"
                          "          [--max_iter n] [--tol t] [--seed s] [--NThreads n]\n"
                          "          [dataset_path]\n";

enum class TYPE
{
    CPU = 0,
    GPU = 1,
    BOTH = 2
};

enum class VERSION
{
    NORMAL = 0,
    UNROLLED = 1,
    SHARED = 2,
    SHARED_UNROLLED = 3
};

struct Config
{
    int NThreads;
    TYPE type;
    VERSION version;
};

static Config config = {
    256, // NThreads
#ifdef ONLY_GPU
    TYPE::GPU, // type
#elif defined(ONLY_CPU)
    TYPE::CPU, // type
#else
    TYPE::BOTH, // type
#endif

#ifdef V_SHARED_UNROLLED
    VERSION::SHARED_UNROLLED // version
#elif defined(V_SHARED)
    VERSION::SHARED // version
#elif defined(V_UNROLLED)
    VERSION::UNROLLED // version
#else
    VERSION::NORMAL // version
#endif
};

static inline std::string get_config_string(const Config *conf)
{
    std::string s;
    s += (conf->type == TYPE::CPU) ? "CPU" : "GPU";
    s += ", ";
    switch (conf->version)
    {
    case VERSION::NORMAL:
        s += "NORMAL";
        break;
    case VERSION::UNROLLED:
        s += "UNROLLED";
        break;
    case VERSION::SHARED:
        s += "SHARED";
        break;
    case VERSION::SHARED_UNROLLED:
        s += "SHARED_UNROLLED";
        break;
    default:
        s += "UNKNOWN";
        break;
    }
    return s;
}

// Inizializza la matrice di appartenenza U (N x K) con valori casuali
// normalizzati riga per riga (ogni punto ha appartenenze che sommano a 1).
// Lo stesso U_init viene passato sia alla versione CPU che a quella GPU,
// cosi' i due run partono dallo stesso punto e i risultati sono confrontabili.
static void init_membership(std::vector<float> &U, int N, int K, unsigned int seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    U.resize((size_t)N * K);
    for (int i = 0; i < N; ++i)
    {
        float *row = &U[(size_t)i * K];
        float sum = 0.0f;
        for (int j = 0; j < K; ++j)
        {
            row[j] = dist(rng);
            sum += row[j];
        }
        for (int j = 0; j < K; ++j)
            row[j] /= sum;
    }
}

// Funzione obiettivo del Fuzzy C-Means:  J = sum_i sum_j u_ij^m * ||x_i - c_j||^2
// Usata solo per validare che CPU e GPU convergano a soluzioni equivalenti.
static double fcm_objective(const float *data, const float *U, const float *C,
                            int N, int D, int K, float m)
{
    double J = 0.0;
    for (int i = 0; i < N; ++i)
    {
        const float *x = data + (size_t)i * D;
        const float *u = U + (size_t)i * K;
        for (int j = 0; j < K; ++j)
        {
            const float *c = C + (size_t)j * D;
            double d2 = 0.0;
            for (int d = 0; d < D; ++d)
            {
                double diff = (double)x[d] - (double)c[d];
                d2 += diff * diff;
            }
            J += std::pow((double)u[j], (double)m) * d2;
        }
    }
    return J;
}

int main(int argc, char **argv)
{
    // Parametri di default
    int N = 0; // 1M punti
    int D = 32;
    int K = 100;
    float m = 2.0f;
    int max_iter = 100;
    float tol = 1e-4f;
    unsigned int seed = 42;
    std::string dataset_path = argc > 1 ? argv[1] : "";

    if (argc <= 1 || std::string(argv[1]) == "--help")
    {
        printf("USAGE: %s [--N points] [--D dimensions] [--K clusters] [--m fuzzifier]\n"
               "          [--max_iter n] [--tol t] [--seed s] [--NThreads n]\n"
               "          [dataset_path]\n",
               argv[0]);
        return 0;
    }

    for (int i = 1; i < argc; ++i)
    {
        if (!strcmp(argv[i], "--N") && i + 1 < argc)
            N = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--D") && i + 1 < argc)
            // this is always 41 but for avoiding unrolling with O3 option we keep it as a parameter,
            D = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--K") && i + 1 < argc)
            K = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--m") && i + 1 < argc)
            m = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--max_iter") && i + 1 < argc)
            max_iter = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tol") && i + 1 < argc)
            tol = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc)
            seed = (unsigned int)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dataset") && i + 1 < argc)
            dataset_path = argv[++i];
        else if (!strcmp(argv[i], "--NThreads") && i + 1 < argc)
        {
            config.NThreads = atoi(argv[++i]);
            printf("NThreads set to %d\n", config.NThreads);
        }
        else if (!strcmp(argv[i], "--type") && i + 1 < argc)
        {
            std::string t = argv[++i];
            if (t == "CPU")
                config.type = TYPE::CPU;
            else if (t == "GPU")
                config.type = TYPE::GPU;
            else if (t == "BOTH")
                config.type = TYPE::BOTH;
            else
            {
                fprintf(stderr, "Error: unknown type '%s'\n", t.c_str());
                return 1;
            }
        }
        else if (!strcmp(argv[i], "--help"))
        {
            printf("USAGE: %s [--N points] [--D dimensions] [--K clusters] [--m fuzzifier]\n"
                   "          [--max_iter n] [--tol t] [--seed s] [--NThreads n]\n",
                   argv[0]);
            return 0;
        }
    }

    // 1) Load dataset from file or generate synthetic blobs
    std::vector<float> data;
    std::vector<int> true_labels;

    if (dataset_path.empty() && N <= 0)
    {
        fprintf(stderr, "Error: please specify a dataset or a number of points N>0\n");
        return 1;
    }

    printf("Loading files: %s\n", dataset_path.c_str());
    auto parser = KDDParser();
    parser.load(dataset_path, true, N);
    DataMatrix dm = parser.to_matrix(/*normalize=*/true, /*log_scale=*/true);
    data.resize((size_t)dm.N * dm.D);
    std::memcpy(data.data(), dm.data, (size_t)dm.N * dm.D * sizeof(float));
    true_labels.resize(dm.N);
    std::memcpy(true_labels.data(), dm.label_ids, (size_t)dm.N * sizeof(int));
    N = dm.N;
    D = dm.D;
    K = (int)parser.label_names.size();
    K = 23; // for the KDD dataset, we know there are 23 unique labels
    printf("=== Fuzzy C-Means: benchmark CPU vs GPU ===\n");
    printf("N=%d  D=%d  K=%d  m=%.2f  max_iter=%d  tol=%g  seed=%u nThreads=%d\n\n",
           N, D, K, m, max_iter, tol, seed, config.NThreads);
    printf("Configuration: %s\n", get_config_string(&config).c_str());

    // 2) Stessa inizializzazione casuale per entrambi i run
    std::vector<float> U_init;
    init_membership(U_init, N, K, seed + 1);

    std::vector<float> U_cpu((size_t)N * K), C_cpu((size_t)K * D);
    std::vector<float> U_gpu((size_t)N * K), C_gpu((size_t)K * D);
    int iters_cpu = 0, iters_gpu = 0;

    // 3) Run CPU
    printf("\n--- CPU (single-thread) ---\n");
    double t_cpu = 0.0f;

    if (config.type == TYPE::CPU || config.type == TYPE::BOTH)
    {
        printf("Executing FCM CPU...\n");
        t_cpu = fcm_cpu_run(data.data(), N, D, K, m, max_iter, tol,
                            U_init.data(), U_cpu.data(), C_cpu.data(), &iters_cpu);
    }
    else
    {
        printf("CPU run skipped (config.type=%d)\n", (int)config.type);
    }

    printf("Iterations: %d\n", iters_cpu);
    printf("CPU:  %.2f ms\n", t_cpu);

    // 4) Run GPU
    if (config.type == TYPE::GPU || config.type == TYPE::BOTH)
    {
        printf("Executing FCM GPU...\n");
    }
    else
    {
        printf("GPU run skipped (config.type=%d)\n", (int)config.type);
    }
    printf("\n--- GPU (CUDA) ---\n");
    double t_gpu = config.type == TYPE::GPU || config.type == TYPE::BOTH
                       ? fcm_gpu_run(data.data(), N, D, K, m, max_iter, tol,
                                     U_init.data(), U_gpu.data(), C_gpu.data(), &iters_gpu, config.NThreads)
                       : 0.0;
    printf("Iterations: %d\n", iters_gpu);
    printf("GPU:  %.2f ms\n", t_gpu);

    // 5) Validazione: la funzione obiettivo deve essere comparabile
    double J_cpu = fcm_objective(data.data(), U_cpu.data(), C_cpu.data(), N, D, K, m);
    double J_gpu = fcm_objective(data.data(), U_gpu.data(), C_gpu.data(), N, D, K, m);

    printf("\n=== Results ===\n");
    printf("Function J (CPU): %f\n", J_cpu);
    printf("Function J (GPU): %f\n", J_gpu);
    printf("Relative difference:        %.6f%%\n",
           100.0 * std::fabs(J_cpu - J_gpu) / J_cpu);
    printf("Speedup (T_cpu / T_gpu):    %.2fx\n", t_cpu / t_gpu);

    // write results to file
    FILE *f = fopen("results.csv", "a");
    if (f)
    {
        // if there is already a file append to it else create the header
        if (ftell(f) == 0)
        {
            fprintf(f, "type,version,n_threads,n_blocks,N,D,K,m,max_iter,tol,seed,T_cpu,T_gpu,J_cpu,J_gpu,diff_rel,speedup\n");
        }

        auto config_str = get_config_string(&config);
        printf("threads=%d, config=%s\n", config.NThreads, config_str.c_str());
        fprintf(f, "%s,%s,%d,%d,%d,%d,%d,%d,%.2f,%d,%g,%u,%.2f,%.2f,%f,%f,%.6f,%.2f\n",
                (config.type == TYPE::CPU ? "CPU" : "GPU"),
                (config.version == VERSION::NORMAL ? "NORMAL" : config.version == VERSION::UNROLLED ? "UNROLLED"
                                                            : config.version == VERSION::SHARED     ? "SHARED"
                                                                                                    : "SHARED_UNROLLED"),
                config.NThreads, 0,
                N, D, K, K, m, max_iter, tol, seed,
                t_cpu, t_gpu, J_cpu, J_gpu,
                100.0 * std::fabs(J_cpu - J_gpu) / J_cpu,
                t_cpu / t_gpu);
    }
    else
    {
        fprintf(stderr, "Error: cannot open results.csv for writing\n");
    }

    return 0;
}
