#pragma once
#include <cuda_runtime.h>
/**
 * @brief c=a+b
 * 
 *
 * 
 * @param a 
 * @param b 
 * @param c 
 * @param N 向量维度
 */
 __global__ void vec_add_cuda_kernel1(int* a, int* b, int *c,int N);
