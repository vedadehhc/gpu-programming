#include <stdio.h>
#include <cuda.h>

void logCudaError(cudaError_t err, int line)
{
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, line);
        exit(EXIT_FAILURE);
    }
}

__host__ __device__ float *getPointer(float *matrix, int numRows, int numCols, int row, int col)
{
    if (row < 0 || row >= numRows)
        return nullptr;
    if (col < 0 || col >= numCols)
        return nullptr;
    return &matrix[row * numCols + col];
}

__global__ void computeSingleEntry(float *d_A, float *d_B, float *d_C, int M, int N, int K) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    float *p_C = getPointer(d_C, M, K, row, col);
    if (p_C == nullptr) return;
    
    *p_C = 0.0f;

    float *p_A;
    float *p_B;
    for (int i = 0; i < N; i++) {
        p_A = getPointer(d_A, M, N, row, i);
        p_B = getPointer(d_B, N, K, i, col);
        if (p_A == nullptr || p_B == nullptr) continue;

        *p_C += (*p_A) * (*p_B); 
    }
}

int main()
{
    int M = 1 << 12;
    int N = 100;
    int K = 1 << 15;
    float A_val = 9.2f;
    float B_val = 4.3f;
    float C_val = A_val * B_val * N;

    // A = M x N, B = N x K, C = M x K

    // allocate host memory
    float *h_A, *h_B, *h_C;
    h_A = (float *)malloc(M * N * sizeof(float));
    h_B = (float *)malloc(N * K * sizeof(float));
    h_C = (float *)malloc(M * K * sizeof(float));

    // allocate device memory
    float *d_A, *d_B, *d_C;
    logCudaError(cudaMalloc(&d_A, M * N * sizeof(float)), __LINE__);
    logCudaError(cudaMalloc(&d_B, N * K * sizeof(float)), __LINE__);
    logCudaError(cudaMalloc(&d_C, M * K * sizeof(float)), __LINE__);

    // initialize h_A, h_B
    float *pMatrix;
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++) {
            pMatrix = getPointer(h_A, M, N, i, j);
            if (pMatrix != nullptr) {
                *pMatrix = A_val;
            } else {
                printf("ERROR initializing A\n");
            }
        }
    }
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < K; j++) {
            pMatrix = getPointer(h_B, N, K, i, j);
            if (pMatrix != nullptr) {
                *pMatrix = B_val;
            } else {
                printf("ERROR initializing B\n");
            }
        }
    }

    // memcpy host to device
    cudaMemcpy(d_A, h_A, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * K * sizeof(float), cudaMemcpyHostToDevice);

    // kernel call
    dim3 gridDim((K + 1)/2, (M + 1)/2, 1);
    dim3 blockDim(2, 2, 1);
    computeSingleEntry<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);

    // memcpy device to host
    cudaMemcpy(h_C, d_C, M * K * sizeof(float), cudaMemcpyDeviceToHost);

    // validate result
    float maxError = 0.0f;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            pMatrix = getPointer(h_C, M, K, i, j);
            if (pMatrix != nullptr) {
                maxError = max(maxError, abs(*pMatrix - C_val));
            } else {
                printf("ERROR reading C\n");
            }
        }
    }
    printf("Max error: %f\n", maxError);

    // free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}