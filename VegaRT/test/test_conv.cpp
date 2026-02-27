#include "layer/vision/conv.hpp"
#include "tensor.hpp"
#include "common.hpp"
#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>
#include "layer/layer_factory.hpp"
#include "utils.hpp"

TEST(test_registry, create_layer_conv_forward) {
    uint32_t batch_size = 1;
    std::vector<vega_rt::TensorSP> inputs(batch_size);
    std::vector<vega_rt::TensorSP> outputs(batch_size);

    const uint32_t kernel_h = 3;
    const uint32_t kernel_w = 3;
    const uint32_t stride_h = 1;
    const uint32_t stride_w = 1;
    const uint32_t kernel_count = 2;
    const uint32_t input_channel = 2;
    const uint32_t output_channel = 2;
    const uint32_t padding_h = 0;
    const uint32_t padding_w = 0;
    const uint32_t groups = 1;
    const bool use_bias = false;


    // 构造input, shape (1,2,4,4)
    for(uint32_t i = 0; i < batch_size; ++i) {
      vega_rt::TensorSP input = vega_rt::TensorCreate(input_channel, 4, 4);
      input->data().slice(0) = arma::fmat("1,2,3,4;"
                                          "5,6,7,8;"
                                          "9,10,11,12;"
                                          "1,2,3,4");
      input->data().slice(1) = arma::fmat("1,2,3,4;"
                                          "2,3,4,5;"
                                          "3,4,5,6;"
                                          "4,5,6,7");
      inputs.at(i) = input;
    }

    // 设置卷积核权重
     // 这属于两核两通道，两个卷积核对应输出的通道数2，每个卷积核两通道对应输入的通道数2
    // 如果要求输出通道数4，那么就需要4核2通道，即4个通道数为2的卷积核，这是不考虑分组卷积的情况下
    // 如果考虑分组卷积，对于input_channel4，output_channel8，分为2组，那么需要8个卷积核，每个卷积核2通道(4/2)
    std::vector<vega_rt::TensorSP> weights(kernel_count);
    for(uint32_t i = 0; i < kernel_count; ++i) {
      vega_rt::TensorSP kernel = vega_rt::TensorCreate(input_channel, kernel_h, kernel_w);
      kernel->data().slice(0) = arma::fmat("1,2,3;"
                                                          "3,2,1;"
                                                          "1,2,3;");
      kernel->data().slice(1) = arma::fmat("1,2,3;"
                                                          "3,2,1;"
                                                          "1,2,3;");
      weights.at(i) = kernel;
    }


    vega_rt::ConvLayer layer(output_channel,
      input_channel,
      kernel_h,
      kernel_w,
      padding_h,
      padding_w,
      stride_h,
      stride_w,
      groups,
      use_bias);
    layer.set_weights(weights);
    layer.Forward(inputs, outputs);
}


TEST(test_registry, create_layer_conv_groups) {
  uint32_t batch_size = 1;
  std::vector<vega_rt::TensorSP> inputs(batch_size);
  std::vector<vega_rt::TensorSP> outputs(batch_size);

  const uint32_t kernel_h = 3;
  const uint32_t kernel_w = 3;
  const uint32_t stride_h = 1;
  const uint32_t stride_w = 1;
  const uint32_t input_channel = 2;
  const uint32_t output_channel = 2;
  const uint32_t padding_h = 0;
  const uint32_t padding_w = 0;
  const uint32_t groups = 2; // 分两组
  const bool use_bias = false;


  // 构造input, shape (1,2,4,4)
  for(uint32_t i = 0; i < batch_size; ++i) {
    vega_rt::TensorSP input = vega_rt::TensorCreate(input_channel, 4, 4);
    input->data().slice(0) = arma::fmat("1,2,3,4;"
                                        "5,6,7,8;"
                                        "9,10,11,12;"
                                        "1,2,3,4");
    input->data().slice(1) = arma::fmat("1,2,3,4;"
                                        "2,3,4,5;"
                                        "3,4,5,6;"
                                        "4,5,6,7");
    inputs.at(i) = input;
  }

  // 普通卷积：卷积核个数为output_channel，卷积核通道数为input_channel
  // 分组卷积：卷积核个数为output_channel，卷积核通道数为input_channel/groups
  const uint32_t kernel_count = output_channel;
  const uint32_t kernel_channels = input_channel / groups;
  std::vector<vega_rt::TensorSP> weights(kernel_count);
  for(uint32_t i = 0; i < kernel_count; ++i) {
    vega_rt::TensorSP kernel = vega_rt::TensorCreate(kernel_channels, kernel_h, kernel_w);
    kernel->data().slice(0) = arma::fmat("1,2,3;"
                                                        "3,2,1;"
                                                        "1,2,3;");
    // kernel->data().slice(1) = arma::fmat("1,2,3;"
    //                                                     "3,2,1;"
    //                                                     "1,2,3;");
    weights.at(i) = kernel;
  }


  vega_rt::ConvLayer layer(output_channel,
    input_channel,
    kernel_h,
    kernel_w,
    padding_h,
    padding_w,
    stride_h,
    stride_w,
    groups,
    use_bias);
  layer.set_weights(weights);
  layer.Forward(inputs, outputs);
}
int main(int argc, char** argv) {
    vega_rt::init_logging(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}