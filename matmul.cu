#include <cstdio>
#include <cuda_runtime.h>

#define N (1 << 14)
#define BLOCK_SIZE 256
#define TILE_SIZE 16
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

struct Matrix
{
    int width;
    int height;
    float *elements;
};

void matMul_CPU(const Matrix A, const Matrix B, Matrix C)
{
    for (int i = 0; i < A.height; ++i)
    {
        for (int j = 0; j < B.width; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < A.width; ++k)
            {
                sum += A.elements[i * A.width + k] * B.elements[k * B.width + j];
            }
            C.elements[i * C.width + j] = sum;
        }
    }
}

void initMatrix(Matrix *M)
{
    for (int i = 0; i < M->width * M->height; ++i)
    {
        M->elements[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

int main(void)
{
    Matrix A = {N, N, nullptr};
    Matrix B = {N, N, nullptr};
    Matrix C = {N, N, nullptr};
    A.elements = (float *)malloc(A.width * A.height * sizeof(float));
    B.elements = (float *)malloc(B.width * B.height * sizeof(float));
    C.elements = (float *)malloc(C.width * C.height * sizeof(float));
    initMatrix(&A);
    initMatrix(&B);
    printf("Starting matrix multiplication on CPU...\n");
    matMul_CPU(A, B, C);
    printf("Matrix multiplication completed on CPU.\n");
    for (int i = 0; i < 5; ++i)
    {
        printf("C[%d][%d] = %f\n", i, i, C.elements[i * C.width + i]);
    }
    free(A.elements);
    free(B.elements);
    free(C.elements);
    return 0;
}