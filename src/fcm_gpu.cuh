#ifndef FCM_GPU_CUH
#define FCM_GPU_CUH

// Same interface as the CPU implementation (fcm_cpu.h), allowing the caller
// (main.cpp) to invoke either implementation interchangeably.
//
// IMPORTANT: this header does not expose any CUDA-specific types (e.g.,
// cudaError_t, kernels, etc.). As a result, main.cpp can be compiled with
// a standard C++ compiler (g++/clang++) without requiring nvcc. Only
// fcm_gpu.cu must be compiled with nvcc, together with the final linking
// step against the CUDA runtime library.
//
// Returns the GPU execution time in milliseconds, measured using CUDA events.
// The reported time includes kernel execution and the associated memory copies,
// but excludes CUDA context initialization overhead.
double fcm_gpu_run(const float *data, int N, int D, int K, float m,
                   int max_iter, float tol,
                   const float *U_init,
                   float *U_out, float *C_out, int *iters_out, int nThreads = 0);

#endif // FCM_GPU_CUH