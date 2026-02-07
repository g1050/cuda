#ifndef MATRIX_OPS_H
#define MATRIX_OPS_H

#include <armadillo>
#include <glog/logging.h>

/**
 * @brief 手动实现矩阵加法
 * 
 * @param A 矩阵A
 * @param B 矩阵B
 * @return 矩阵C
 */
arma::mat manual_matrix_add(const arma::mat& A, const arma::mat& B) {
    // 检查矩阵维度是否匹配
    if (A.n_rows != B.n_rows || A.n_cols != B.n_cols) {
        LOG(ERROR) << "矩阵维度不匹配！A: " << A.n_rows << "x" << A.n_cols 
                   << ", B: " << B.n_rows << "x" << B.n_cols;
        return arma::mat();  // 返回空矩阵
    }
    
    // 创建结果矩阵，大小与输入矩阵相同
    arma::mat C(A.n_rows, A.n_cols);
    
    // 手动遍历每个元素并相加
    for (size_t i = 0; i < A.n_rows; ++i) {
        for (size_t j = 0; j < A.n_cols; ++j) {
            C(i, j) = A(i, j) + B(i, j);
        }
    }
    
    return C;
}

#endif // MATRIX_OPS_H
