#include "graph.hpp"
#include "tensor.hpp"
#include "common.hpp"
#include "operator.hpp"
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include "utils.hpp"

TEST(OperatorTest, TestGraphInit) {
    std::string param_path = "/root/autodl-tmp/cuda/VegaRT/python/model_file/model.pnnx.param";
    std::string bin_path = "/root/autodl-tmp/cuda/VegaRT/python/model_file/model.pnnx.bin";
    
    // std::string param_path_2 = "/root/autodl-tmp/cuda/download/kuiperdatawhale/course3/model_file/test_linear.pnnx.param";
    // std::string bin_path_2 = "/root/autodl-tmp/cuda/download/kuiperdatawhale/course3/model_file/test_linear.pnnx.bin";
    std::unique_ptr<vega_rt::Graph> graph = std::make_unique<vega_rt::Graph>(param_path, bin_path);
    ASSERT_EQ(graph->Init(), vega_rt::VegaError::Success);
    const auto &operators = graph->operators();
    for (const auto &operator_ : operators) {
        LOG(INFO) << "op name: " << operator_->name_ << " type: " << operator_->type_;
        LOG(INFO) << "attribute:";
        for (const auto &[name, attribute_] : operator_->attrs_) {
        LOG(WARNING) << name << " type: " << vega_rt::DataTypeStr(attribute_->type_)
                    << " shape: " << vega_rt::ShapeStr(attribute_->shape_);
        const auto &weight_data = attribute_->weight_data_;
        ASSERT_EQ(weight_data.empty(), false); // 判断权重是否为空
        }
        LOG(INFO) << "inputs: ";
        for (const auto &input : operator_->input_operands_list_) {
            LOG(WARNING) << "name: " << input->name_
                    << " shape: " << vega_rt::ShapeStr(input->shapes_);
        }

        LOG(INFO) << "outputs: ";
        for (const auto &output : operator_->output_names_) {
            LOG(WARNING) << "name: " << output;
        }
        LOG(INFO) << "params: ";
        for (const auto &[name, param] : operator_->params_) {
            LOG(WARNING) << "name: " << name << " type: " << vega_rt::ParameterTypeStr(param->type_) ;
        }

        LOG(INFO) << "--------------------------------------";
    }
}
TEST(test_ir, pnnx_graph_test_linear) {
    std::string bin_path("/root/autodl-tmp/cuda/VegaRT/python/model_file/model.pnnx.bin");
    std::string param_path("/root/autodl-tmp/cuda/VegaRT/python/model_file/model.pnnx.param");
    std::unique_ptr<vega_rt::Graph> graph = std::make_unique<vega_rt::Graph>(param_path, bin_path);
    ASSERT_EQ(graph->Init(), vega_rt::VegaError::Success);
    const auto &operators = graph->operators();
    for (const auto &operator_ : operators) {
      if (operator_->name_ == "linear") {
        const auto &params = operator_->params_;
        ASSERT_EQ(params.size(), 3);
          /////////////////////////////////
        ASSERT_EQ(params.count("bias"), 1);
        vega_rt::ParameterSP parameter_bool = params.at("bias");
        ASSERT_NE(parameter_bool, nullptr);
        ASSERT_EQ((dynamic_cast<vega_rt::ParameterBool*>(parameter_bool.get())->value_),
                  true);
        /////////////////////////////////
        ASSERT_EQ(params.count("in_features"), 1);
        vega_rt::ParameterSP parameter_in_features = params.at("in_features");
        ASSERT_NE(parameter_in_features, nullptr);
        ASSERT_EQ(
            (dynamic_cast<vega_rt::ParameterInt*>(parameter_in_features.get())->value_),
            32);
  
        /////////////////////////////////
        ASSERT_EQ(params.count("out_features"), 1);
        vega_rt::ParameterSP parameter_out_features = params.at("out_features");
        ASSERT_NE(parameter_out_features, nullptr);
        ASSERT_EQ(
            (dynamic_cast<vega_rt::ParameterInt*>(parameter_out_features.get())->value_),
            128);
      }
    }
}

TEST(OperatorTest, TestGraphBuild) {
    std::string bin_path("/root/autodl-tmp/cuda/VegaRT/python/model_file/resnet18.pnnx.bin");
    std::string param_path("/root/autodl-tmp/cuda/VegaRT/python/model_file/resnet18.pnnx.param");
    std::unique_ptr<vega_rt::Graph> graph = std::make_unique<vega_rt::Graph>(param_path, bin_path);
    ASSERT_EQ(graph->Init(), vega_rt::VegaError::Success);
    ASSERT_EQ(graph->Build("pnnx_input_0", "pnnx_output_0"), vega_rt::VegaError::Success);

    auto topo_queues = graph->get_topo_queues();
    for (const auto &operator_ : topo_queues) {
      LOG(WARNING) << "operator name: " << operator_->name_ << "\t\t type: " << operator_->type_;
    }
}

int main(int argc, char** argv) {
    vega_rt::init_logging(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}