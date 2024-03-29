#include <stdio.h>
#include <cuda.h>
#include <time.h>

void logCudaError(cudaError_t err, int line)
{
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, line);
        exit(EXIT_FAILURE);
    }
}

int timeCheckpoint(clock_t &t)
{
    logCudaError(cudaDeviceSynchronize(), __LINE__);
    int msec = (clock() - t) * 1000 / CLOCKS_PER_SEC;
    t = clock();
    return msec;
}

__host__ __device__ float *getPointer(float *matrix, int numRows, int numCols, int row, int col)
{
    if (row < 0 || row >= numRows)
        return nullptr;
    if (col < 0 || col >= numCols)
        return nullptr;
    return (matrix + row * numCols + col);
}

__global__ void computeSingleEntry(float *d_A, float *d_B, float *d_C, int M, int N, int K) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    float *p_C = getPointer(d_C, M, K, row, col);
    if (p_C == nullptr) return;
    
    float res = 0.0f;

    float *p_A;
    float *p_B;
    for (int i = 0; i < N; i++) {
        p_A = getPointer(d_A, M, N, row, i);
        p_B = getPointer(d_B, N, K, i, col);
        // if (p_A == nullptr || p_B == nullptr) continue;

        res += (*p_A) * (*p_B); 
    }
    *p_C = res;
}

int main()
{
    clock_t start = clock(), t = clock();

    int M = 5000;
    int N = 10000;
    int K = 5000;
    float A_val = 1.0f;
    float B_val = 2.0f;
    float C_val = A_val * B_val * N;

    // A = M x N, B = N x K, C = M x K

    // allocate host memory
    float *h_A, *h_B, *h_C;
    h_A = (float *)malloc(M * N * sizeof(float));
    h_B = (float *)malloc(N * K * sizeof(float));
    h_C = (float *)malloc(M * K * sizeof(float));

    printf("Host allocation completed \t %d ms\n", timeCheckpoint(t));

    // allocate device memory
    float *d_A, *d_B, *d_C;
    logCudaError(cudaMalloc(&d_A, M * N * sizeof(float)), __LINE__);
    logCudaError(cudaMalloc(&d_B, N * K * sizeof(float)), __LINE__);
    logCudaError(cudaMalloc(&d_C, M * K * sizeof(float)), __LINE__);

    printf("Device allocation completed \t %d ms\n", timeCheckpoint(t));

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

    printf("Host matrix initialization completed \t %d ms\n", timeCheckpoint(t));

    // memcpy host to device
    cudaMemcpy(d_A, h_A, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * K * sizeof(float), cudaMemcpyHostToDevice);

    printf("Host to Device Memcpy completed \t %d ms\n", timeCheckpoint(t));

    // kernel call
    dim3 gridDim((K + 1)/2, (M + 1)/2, 1);
    dim3 blockDim(2, 2, 1);

    computeSingleEntry<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);

    // wait for sync for timing
    printf("Kernel call completed \t %d ms\n", timeCheckpoint(t));

    // memcpy device to host
    cudaMemcpy(h_C, d_C, M * K * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Device to Host Memcpy completed \t %d ms\n", timeCheckpoint(t));

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
    
    printf("Validation completed \t %d ms\n", timeCheckpoint(t));

    // free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    printf("Memory free completed \t %d ms\n", timeCheckpoint(t));

    printf("\nTOTAL TIME: %d ms\n", timeCheckpoint(start));

    return 0;
}