#include "softmax_cpu.h"
#include "vec_add.h"
#include <iostream>
#include <cstdlib>
template<typename T>
void print_output(T *arr, int N, int C) {
    
    for (int i = 0; i < N; i++) {
        T sum = 0;
        for (int j = 0; j < C; j++) {
            std::cout << arr[i * C + j] << " ";
            sum += arr[i * C + j];
        }
        // std::cout << "\nsum: " << sum << std::endl;
        std::cout << std::endl;
    }
}

void test_softmax_cpu() {
    int N = 10;
    int C = 10;
    float* input = new float[N * C];
    float* output = new float[N * C];
    for (int i = 0; i < N * C; i++) {
        input[i] = rand() % 100;
    }
    softmax_cpu(output, input, N, C);
    print_output<float>(output, N, C);
}

void test_vec_add(){
    int N = 10;
    int *A = new int[N];
    int *B = new int[N];
    int *C = new int[N];
    int BLOCK_SIZE = 16;
    for (int i = 0; i < N; i++) {
        A[i] = i;
        B[i] = i;
    }
    int *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, N * sizeof(int));
    cudaMalloc((void **)&d_B, N * sizeof(int));
    cudaMalloc((void **)&d_C, N * sizeof(int));
    cudaMemcpy(d_A, A, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * sizeof(int), cudaMemcpyHostToDevice);
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    vec_add_cuda_kernel1<<<numBlocks, BLOCK_SIZE>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    cudaMemcpy(C, d_C, N * sizeof(int), cudaMemcpyDeviceToHost);
    print_output<int>(C, 1, N);
}

int main() {
    test_vec_add();
    return 0;
}