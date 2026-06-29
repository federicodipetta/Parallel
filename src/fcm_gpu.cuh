#ifndef FCM_GPU_CUH
#define FCM_GPU_CUH

// Stessa interfaccia della versione CPU (fcm_cpu.h), cosi il chiamante
// (main.cpp) puo usare le due funzioni in modo intercambiabile.
//
// IMPORTANTE: questo header non include alcun tipo CUDA (no cudaError_t,
// no kernel, ecc). Per questo main.cpp puo essere compilato direttamente
// con g++, senza bisogno di nvcc: nvcc serve solo per compilare fcm_gpu.cu
// e per il link finale (necessario per risolvere i simboli di cudart).
//
// Ritorna il tempo di calcolo GPU in millisecondi (misurato con cudaEvent,
// quindi solo kernel + memcpy associati, non l'overhead di inizializzazione
// del contesto CUDA).
double fcm_gpu_run(const float* data, int N, int D, int K, float m,
                    int max_iter, float tol,
                    const float* U_init,
                    float* U_out, float* C_out, int* iters_out);

#endif // FCM_GPU_CUH
