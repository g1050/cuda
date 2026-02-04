#include "vec_add.h"

__global__ void vec_add_cuda_kernel1(int* a, int* b, int *c,int N) {
    int i = blockIdx.x * blockDim.x+threadIdx.x;
    if (i < N) {
        c[i] = a[i] + b[i];
    }
}