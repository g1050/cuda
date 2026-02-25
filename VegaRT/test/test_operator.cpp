#include "layer/vision/conv.hpp"
#include "tensor.hpp"
#include "common.hpp"
#include <gtest/gtest.h>
#include <iostream>
#include "layer/layer_factory.hpp"
#include "utils.hpp"
static vega_rt::LayerRegisterer::CreateRegistry *RegistryGlobal() {
    static vega_rt::LayerRegisterer::CreateRegistry *kRegistry = new vega_rt::LayerRegisterer::CreateRegistry();
    CHECK(kRegistry != nullptr) << "Global layer register init failed!";
    return kRegistry;
}

TEST(test_registry, registry1) {
    // 同一个实例对象
    vega_rt::LayerRegisterer::CreateRegistry *registry1 = RegistryGlobal();
    vega_rt::LayerRegisterer::CreateRegistry *registry2 = RegistryGlobal();
    ASSERT_EQ(registry1, registry2);
}

vega_rt::VegaError MyTestCreator(
    const vega_rt::OperatorSP &op,
    vega_rt::LayerSP &layer) {

  layer = std::make_shared<vega_rt::Layer>("test_layer");
  return vega_rt::VegaError::Success;
}

TEST(test_registry, registry2) {
    vega_rt::LayerRegisterer::CreateRegistry registry1 = vega_rt::LayerRegisterer::Registry();
    vega_rt::LayerRegisterer::CreateRegistry registry2 = vega_rt::LayerRegisterer::Registry();
    vega_rt::LayerRegisterer::RegisterCreator("test_type", MyTestCreator);
    vega_rt::LayerRegisterer::RegisterCreator("test_type2", MyTestCreator);

    vega_rt::LayerRegisterer::CreateRegistry registry3 = vega_rt::LayerRegisterer::Registry();
    ASSERT_EQ(registry1, registry2);
    // ASSERT_EQ(registry3.size(), 3); //算子在不断增加,所以不能这样判断
    ASSERT_NE(registry3.find("test_type"), registry3.end());
  }

TEST(test_registry, create_layer) {
    // 注册了一个test_type_1算子
    vega_rt::LayerRegisterer::RegisterCreator("test_type_1", MyTestCreator);
    vega_rt::OperatorSP op = std::make_shared<vega_rt::Operator>();
    op->type_ = "test_type_1";
    vega_rt::LayerSP layer;
    ASSERT_EQ(layer, nullptr);
    layer = vega_rt::LayerRegisterer::CreateLayer(op);
    ASSERT_NE(layer, nullptr);
}

TEST(test_registry, create_layer_util) {
    // 注册算子
    vega_rt::LayerRegistererWrapper kReluGetInstance("test_type_2", MyTestCreator);
    vega_rt::OperatorSP op = std::make_shared<vega_rt::Operator>();
    op->type_ = "test_type_2";
    vega_rt::LayerSP layer;
    ASSERT_EQ(layer, nullptr);
    // 用operator创建算子
    layer = vega_rt::LayerRegisterer::CreateLayer(op);
    ASSERT_NE(layer, nullptr);
}

TEST(test_registry, create_layer_reluforward) {
    vega_rt::OperatorSP op = std::make_shared<vega_rt::Operator>();
    op->type_ = "nn.ReLU";
    vega_rt::LayerSP layer;
    ASSERT_EQ(layer, nullptr);
    layer = vega_rt::LayerRegisterer::CreateLayer(op);
    ASSERT_NE(layer, nullptr);
  
    vega_rt::TensorSP input_tensor = std::make_shared<vega_rt::Tensor<float>>(3, 4, 4);
    input_tensor->Rand(); // 随机初始化
    input_tensor->data() -= 0.5f; // arma::fcube - 0.5f
  
    LOG(INFO) << input_tensor->data();
  
    std::vector<vega_rt::TensorSP> inputs(1); // batch size
    std::vector<vega_rt::TensorSP> outputs(1);
    inputs.at(0) = input_tensor;
    LOG(ERROR) << "inputTensor empty: " << input_tensor->empty();
    LOG(ERROR) << "inputTensor shape: " << vega_rt::ShapeStr(input_tensor->shapes());
    layer->Forward(inputs, outputs);
    for (const auto &output : outputs) {
      output->Show();
    }
}

TEST(test_registry, create_layer_sigmoid_forward) {
    std::shared_ptr<vega_rt::Operator> op = std::make_shared<vega_rt::Operator>();
    op->type_ = "nn.Sigmoid";
    std::shared_ptr<vega_rt::Layer> layer;
    ASSERT_EQ(layer, nullptr);
    layer = vega_rt::LayerRegisterer::CreateLayer(op);
    ASSERT_NE(layer, nullptr);
  
    vega_rt::TensorSP input_tensor = std::make_shared<vega_rt::Tensor<float>>(3, 4, 4);
    input_tensor->Rand();
  
    std::vector<vega_rt::TensorSP> inputs(1);
    std::vector<vega_rt::TensorSP> outputs(1);
    inputs.at(0) = input_tensor;
    layer->Forward(inputs, outputs);
  
    ASSERT_EQ(outputs.size(), 1);
    vega_rt::TensorSP output_tensor = outputs.front();
    ASSERT_EQ(output_tensor->empty(), false);
    ASSERT_EQ(output_tensor->size(), input_tensor->size());
  
    uint32_t size = output_tensor->size();
    // 评价sigmoid的计算结果是否正确
    for (uint32_t i = 0; i < size; ++i) {
      float input_value = input_tensor->index(i);
      float output_value = output_tensor->index(i);
      ASSERT_EQ(output_value, 1 / (1.f + expf(-input_value)));
    }
}

TEST(test_registry, create_layer_maxpooling_forward) {
  std::shared_ptr<vega_rt::Operator> op = std::make_shared<vega_rt::Operator>();
  op->type_ = "nn.MaxPool2d";
  std::vector<int> strides{2, 2};
  std::shared_ptr<vega_rt::ParameterIntArray> stride_param = std::make_shared<vega_rt::ParameterIntArray>(strides);
  op->params_.insert({"stride", stride_param});

  std::vector<int> kernel{2, 2};
  std::shared_ptr<vega_rt::ParameterIntArray> kernel_param = std::make_shared<vega_rt::ParameterIntArray>(kernel);
  op->params_.insert({"kernel_size", kernel_param});

  std::vector<int> paddings{1, 1};
  std::shared_ptr<vega_rt::ParameterIntArray> padding_param = std::make_shared<vega_rt::ParameterIntArray>(paddings);
  op->params_.insert({"padding", padding_param});

  std::shared_ptr<vega_rt::Layer> layer;
  layer = vega_rt::LayerRegisterer::CreateLayer(op);
  ASSERT_NE(layer, nullptr);

  // 设置输入、输出Tensor
  vega_rt::TensorSP tensor = vega_rt::TensorCreate(1, 4, 4);
  arma::fmat input = arma::fmat("1,2,3,4;"
                                "2,3,4,5;"
                                "3,4,5,6;"
                                "4,5,6,7");
  tensor->data().slice(0) = input;
  std::vector<vega_rt::TensorSP> inputs(1);
  inputs.at(0) = tensor;
  std::vector<vega_rt::TensorSP> outputs(1);
  layer->Forward(inputs, outputs);

  for (const auto &output : outputs) {
    output->Show();
  }

}

TEST(test_registry, create_layer_conv_forward) {
    uint32_t batch_size = 1;
    std::vector<vega_rt::TensorSP> inputs(batch_size);
    std::vector<vega_rt::TensorSP> outputs(batch_size);

    const uint32_t kernel_h = 3;
    const uint32_t kernel_w = 3;
    const uint32_t stride_h = 1;
    const uint32_t stride_w = 1;
    const uint32_t kernel_count = 2;
    const uint32_t input_channel = 3;
    const uint32_t output_channel = 2;
    const uint32_t padding_h = 1;
    const uint32_t padding_w = 1;
    const uint32_t groups = 1;
    const bool use_bias = true;
    vega_rt::ConvLayer layer(output_channel,
      input_channel,
      kernel_h,
      kernel_w,
      stride_h,
      stride_w,
      padding_h,
      padding_w,
      groups,
      use_bias);
    layer.Forward(inputs, outputs);
}

int main(int argc, char** argv) {
    vega_rt::init_logging(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}