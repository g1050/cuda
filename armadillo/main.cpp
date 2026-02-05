#include <armadillo>
#include <iostream>
#include <glog/logging.h>
#include "matrix_ops.h"

int main(int argc, char** argv) {
    
    google::InitGoogleLogging(argv[0]);
    
    FLAGS_logtostderr = 1;        // 所有级别都打到终端
    FLAGS_minloglevel = 0;       // 0=INFO, 1=WARNING, 2=ERROR, 3=FATAL
    FLAGS_colorlogtostderr = 1;  // 可选：彩色输出（新版本支持）
    
    LOG(INFO) << "Hello glog!";
    LOG(WARNING) << "This is a warning";
    LOG(ERROR) << "This is an error";
    
    arma::mat A = {{1, 2},
                  {3, 4}};

    arma::mat B = {{5, 6},
                  {7, 8}};

    // 使用 Armadillo 框架的运算符加法
    arma::mat C1 = A + B;
    
    // 使用手动实现的矩阵加法
    arma::mat C2 = manual_matrix_add(A, B);
    
    std::cout << "A =\n" << A << std::endl;
    std::cout << "B =\n" << B << std::endl;
    std::cout << "A + B (Armadillo运算符) =\n" << C1 << std::endl;
    std::cout << "A + B (手动实现) =\n" << C2 << std::endl;

    return 0;
}
