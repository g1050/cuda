#include "softmax_cpu.h"
void softmax_cpu(float* output, float* input, int N,int C) {
    for (int i = 0; i < N; i++) {
        float max_val = input[i * C];
        for (int j = 1; j < C; j++) {
            max_val = std::max(max_val, input[i * C + j]);
        }
        float sum = 0;
        for (int j = 0; j < C; j++) {
            sum += std::exp(input[i * C + j] - max_val);
        }
        for (int j = 0; j < C; j++) {
            output[i * C + j] = std::exp(input[i * C + j] - max_val) / sum;
        }
    }
}