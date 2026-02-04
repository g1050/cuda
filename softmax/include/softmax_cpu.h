#pragma once
#include <iostream>
#include <cmath>
#include <algorithm>

/**
 * @brief softmax cpu实现,N*C的矩阵，N行C列
 * 
 *
 * 
 * @param output 
 * @param input 
 * @param N 向量个数
 * @param C 向量维度
 */
void softmax_cpu(float* output, float* input, int N,int C);