#include <gtest/gtest.h>
#include <armadillo>
#include <glog/logging.h>
#include "matrix_ops.h"

// 测试环境设置
class MatrixOpsTestEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        // 初始化 glog（测试环境）
        FLAGS_logtostderr = 1;
        FLAGS_minloglevel = 3;  // 只显示 FATAL，减少测试输出
        google::InitGoogleLogging("test_matrix_ops");
    }
    
    void TearDown() override {
        google::ShutdownGoogleLogging();
    }
};

// 测试正常矩阵加法
TEST(MatrixOpsTest, NormalAddition) {
    arma::mat A = {{1, 2},
                   {3, 4}};
    
    arma::mat B = {{5, 6},
                   {7, 8}};
    
    arma::mat expected = {{6, 8},
                          {10, 12}};
    
    arma::mat result = manual_matrix_add(A, B);
    
    // 使用 Armadillo 的近似相等比较（考虑浮点数精度）
    EXPECT_TRUE(arma::approx_equal(result, expected, "absdiff", 1e-10));
}

// 测试与 Armadillo 运算符结果一致性
TEST(MatrixOpsTest, ConsistencyWithArmadillo) {
    arma::mat A = {{1.5, 2.3},
                   {3.7, 4.1}};
    
    arma::mat B = {{5.2, 6.8},
                   {7.4, 8.9}};
    
    arma::mat manual_result = manual_matrix_add(A, B);
    arma::mat armadillo_result = A + B;
    
    // 验证手动实现与 Armadillo 运算符结果一致
    EXPECT_TRUE(arma::approx_equal(manual_result, armadillo_result, "absdiff", 1e-10));
}

// 测试不同大小的矩阵
TEST(MatrixOpsTest, DifferentSizes) {
    arma::mat A = {{1, 2, 3},
                   {4, 5, 6}};
    
    arma::mat B = {{7, 8, 9},
                   {10, 11, 12}};
    
    arma::mat expected = {{8, 10, 12},
                          {14, 16, 18}};
    
    arma::mat result = manual_matrix_add(A, B);
    
    EXPECT_TRUE(arma::approx_equal(result, expected, "absdiff", 1e-10));
}

// 测试维度不匹配的情况
TEST(MatrixOpsTest, DimensionMismatch) {
    arma::mat A = {{1, 2},
                   {3, 4}};
    
    arma::mat B = {{5, 6, 7},
                   {8, 9, 10}};
    
    arma::mat result = manual_matrix_add(A, B);
    
    // 维度不匹配时应返回空矩阵
    EXPECT_EQ(result.n_rows, 0);
    EXPECT_EQ(result.n_cols, 0);
}

// 测试单元素矩阵
TEST(MatrixOpsTest, SingleElement) {
    arma::mat A(1, 1);
    A(0, 0) = 42;
    
    arma::mat B(1, 1);
    B(0, 0) = 58;
    
    arma::mat result = manual_matrix_add(A, B);
    arma::mat expected(1, 1);
    expected(0, 0) = 100;
    
    EXPECT_TRUE(arma::approx_equal(result, expected, "absdiff", 1e-10));
}

// 测试零矩阵
TEST(MatrixOpsTest, ZeroMatrix) {
    arma::mat A = {{1, 2},
                   {3, 4}};
    
    arma::mat B = arma::zeros<arma::mat>(2, 2);
    
    arma::mat result = manual_matrix_add(A, B);
    
    // A + 0 应该等于 A
    EXPECT_TRUE(arma::approx_equal(result, A, "absdiff", 1e-10));
}

// 测试负数
TEST(MatrixOpsTest, NegativeNumbers) {
    arma::mat A = {{-1, -2},
                   {-3, -4}};
    
    arma::mat B = {{5, 6},
                   {7, 8}};
    
    // arma::mat expected = {{4, 4},
    //                       {4, 4}};
    arma::mat expected = {{4, 4},
                            {4, 5}};
    arma::mat result = manual_matrix_add(A, B);
    
    EXPECT_TRUE(arma::approx_equal(result, expected, "absdiff", 1e-10));
}

// 测试浮点数精度
TEST(MatrixOpsTest, FloatingPointPrecision) {
    arma::mat A = {{0.1, 0.2},
                   {0.3, 0.4}};
    
    arma::mat B = {{0.5, 0.6},
                   {0.7, 0.8}};
    
    arma::mat expected = {{0.6, 0.8},
                          {1.0, 1.2}};
    
    arma::mat result = manual_matrix_add(A, B);
    
    EXPECT_TRUE(arma::approx_equal(result, expected, "absdiff", 1e-10));
}

// 注册测试环境
::testing::Environment* const test_env = ::testing::AddGlobalTestEnvironment(new MatrixOpsTestEnvironment);
