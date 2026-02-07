#include "tensor.hpp"
#include "common.hpp"
#include <gtest/gtest.h>
#include <iostream>

TEST(TensorTest, TestTensorInit) {
    vega_rt::Tensor<float> tensor(10);
    vega_rt::Tensor<float> tensor2(10, 20);
    vega_rt::Tensor<float> tensor3(10, 20, 30);

    auto shape = tensor3.raw_shape();
    EXPECT_EQ(shape.size(), 3);
    EXPECT_EQ(shape[0], 10);
    EXPECT_EQ(shape[1], 20);
    EXPECT_EQ(shape[2], 30);

   
}

TEST(TensorTest, TestTensorFill) {
    vega_rt::Tensor<float> tensor4(4, 4, 9);
    auto shape4 = tensor4.raw_shape();
    tensor4.show_shape();
    int N = tensor4.size();
    LOG(WARNING) << "N: " << N;
    std::vector<float> values(N);
    for(int i = 0; i < N; i++){
        values[i] = i+1;
    }
    tensor4.Fill(values, true);
    EXPECT_EQ(tensor4.at(3,2,4), values[3 * 4 * 9 + 2 * 9 + 4]);
}

int main(int argc, char** argv) {
    vega_rt::init_logging(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}