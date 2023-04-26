#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <stdlib.h>

const int LENGTH = 20000000;

// 单block多threads
__global__ void vector_add1(float *out, float *a, float *b, int n) 
{
    int tid = threadIdx.x;
    int stride = blockDim.x;
    for (int i = tid; i < n; i += stride)
    {
        out[i] = a[i] + b[i];
    }
}

// 多blocks多threads
__global__ void vector_add2(float *out, float *a, float *b, int n) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < n; i += stride) {
        out[i] = a[i] + b[i];
    }
}

// 共享内存
__global__ void vector_add3(float *out, float *a, float *b, int n) 
{
    __shared__ float temp[256];
    int tid = threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i+= stride) {
        temp[tid] = a[i] + b[i];
        __syncthreads();
        
        out[i] = temp[tid];
    }
}

int correctness(float *serial, float *paralell) {
    for (int i = 0; i < LENGTH; i++) {
        if (serial[i] != paralell[i])
            return 0;
    }
    return 1;
}

int main()
{
    float *a, *b, *out, *out1, *out2, *out3;
    float *d_a, *d_b, *d_out1, *d_out2, *d_out3; 
    printf("Array Length is: %d\n", LENGTH);

    //===================步骤1===================
    // Allocate memory on CPU
    a = (float*)malloc(sizeof(float) * LENGTH);
    b = (float*)malloc(sizeof(float) * LENGTH);
    out = (float*)malloc(sizeof(float) * LENGTH);
    out1 = (float*)malloc(sizeof(float) * LENGTH);
    out2 = (float*)malloc(sizeof(float) * LENGTH);
    out3 = (float*)malloc(sizeof(float) * LENGTH);
 
    // data initializtion
    for (int i = 0; i < LENGTH; i++) {
        a[i] = i % (LENGTH - 2) - 5.0;
        b[i] = i % (LENGTH + 2) + 5.0;
    }
    clock_t start0 = clock();
    for (int i = 0; i < LENGTH; i++) {
        out[i] = a[i] + b[i];
    }
    clock_t finish0 = clock();
    printf("serial running: %f milliseconds\n", (double)(finish0 - start0) / 1000);

    //===================步骤1===================
    // Allocate memory on GPU
    cudaMalloc((void**)&d_a, sizeof(float) * LENGTH);
    cudaMalloc((void**)&d_b, sizeof(float) * LENGTH);
    cudaMalloc((void**)&d_out1, sizeof(float) * LENGTH);
    cudaMalloc((void**)&d_out2, sizeof(float) * LENGTH);
    cudaMalloc((void**)&d_out3, sizeof(float) * LENGTH);
 
    //===================步骤2===================
    // copy operator to GPU
    cudaMemcpy(d_a, a, sizeof(float) * LENGTH, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * LENGTH, cudaMemcpyHostToDevice);
    
    //===================步骤3===================
    // GPU do the work, CPU waits
    cudaEvent_t start, stop;
    float time = 0.0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    // 单block多threads
    cudaEventRecord(start, 0);
    vector_add1<<<1,256>>>(d_out1, d_a, d_b, LENGTH);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("parallel running(one block multiple threads): %f milliseconds\n", time);
    

    // 多blocks多threads
    cudaEventRecord(start, 0);
    vector_add1<<<16,256>>>(d_out2, d_a, d_b, LENGTH);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("parallel running(multiple blocks multiple threads): %f milliseconds\n", time);

    // 共享内存
    cudaEventRecord(start, 0);
    vector_add3<<<16,256>>>(d_out3, d_a, d_b, LENGTH);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("parallel running(shared memory): %f milliseconds\n", time);

    //===================步骤4===================
    // Get results from the GPU
    cudaMemcpy(out1, d_out1, sizeof(float) * LENGTH, cudaMemcpyDeviceToHost);
    cudaMemcpy(out2, d_out2, sizeof(float) * LENGTH, cudaMemcpyDeviceToHost);
    cudaMemcpy(out3, d_out3, sizeof(float) * LENGTH, cudaMemcpyDeviceToHost);
    printf("correctness(one block multiple threads): %d\n", correctness(out, out1));
    printf("correctness(multiple blocks multiple threads): %d\n", correctness(out, out2));
    printf("correctness(shared memory): %d\n", correctness(out, out3));
    
    //===================步骤5===================
    // Free the memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out1);
    cudaFree(d_out2);
    cudaFree(d_out3);

    free(a);
    free(b);
    free(out);
    free(out1);
    free(out2);
    free(out3);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}