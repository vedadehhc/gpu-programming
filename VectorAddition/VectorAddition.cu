#include <stdio.h>
#include <cuda.h>

__global__ void vecAdd(float *d_A, float *d_B, float *d_C, int N)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; 
    if (i < N) {
        d_C[i] = d_A[i] + d_B[i];
    }
}

void logCudaError(cudaError_t err, int line)
{
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, line);
        exit(EXIT_FAILURE);
    }
}

int main()
{
    int N = 1 << 28;
    uint64_t vectorSize = N * sizeof(float);
    float A_val = 1.0f;
    float B_val = 2.0f;
    float C_val = 3.0f;

    // host vectors
    printf("allocating host vectors\n");
    float *h_A, *h_B, *h_C;
    h_A = (float *)malloc(vectorSize);
    h_B = (float *)malloc(vectorSize);
    h_C = (float *)malloc(vectorSize);
    for (int i = 0; i < N; i++)
    {
        h_A[i] = A_val;
        h_B[i] = B_val;
    }

    // device vectors
    printf("allocating device vectors\n");
    float *d_A, *d_B, *d_C;

    // allocate device memory
    cudaError_t d_A_mallocError = cudaMalloc(&d_A, vectorSize);
    logCudaError(d_A_mallocError, __LINE__);

    cudaError_t d_B_mallocError = cudaMalloc(&d_B, vectorSize);
    logCudaError(d_B_mallocError, __LINE__);

    cudaError_t d_C_mallocError = cudaMalloc(&d_C, vectorSize);
    logCudaError(d_C_mallocError, __LINE__);

    // copy memory from host to device
    printf("memcpy host to device\n");
    cudaError_t A_memcpyError = cudaMemcpy(d_A, h_A, vectorSize, cudaMemcpyHostToDevice);
    logCudaError(A_memcpyError, __LINE__);

    cudaError_t B_memcpyError = cudaMemcpy(d_B, h_B, vectorSize, cudaMemcpyHostToDevice);
    logCudaError(B_memcpyError, __LINE__);

    // kernel call
    printf("kernel call\n");
    vecAdd<<<(N + 255) / 256, 256>>>(d_A, d_B, d_C, N);

    // copy result from device to host
    printf("memcpy device to host\n");
    cudaError_t C_memcpyError = cudaMemcpy(h_C, d_C, vectorSize, cudaMemcpyDeviceToHost);
    logCudaError(C_memcpyError, __LINE__);

    // check result
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
    {
        maxError = max(maxError, abs(h_C[i] - C_val));
    }
    printf("Max error: %f\n", maxError);

    // free device memory
    printf("free device memory\n");
    logCudaError(cudaFree(d_A), __LINE__);
    logCudaError(cudaFree(d_B), __LINE__);
    logCudaError(cudaFree(d_C), __LINE__);

    // free host memory
    printf("free host memory\n");
    free(h_A);
    free(h_B);
    free(h_C);
}