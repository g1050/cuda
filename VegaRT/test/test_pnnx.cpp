#include "common.hpp"
#include "pnnx/ir.h"
#include <gtest/gtest.h>
#include <iostream>

TEST(test_pnnx, pnnx_graph_ops) {
    std::string bin_path("/root/autodl-tmp/cuda/VegaRT/python/model_file/model.pnnx.bin");
    std::string param_path("/root/autodl-tmp/cuda/VegaRT/python/model_file/model.pnnx.param");
    std::unique_ptr<pnnx::Graph> graph = std::make_unique<pnnx::Graph>();
    int load_result = graph->load(param_path, bin_path);
    ASSERT_EQ(load_result, 0);
    const auto &ops = graph->ops;
    LOG(WARNING) << "ops size: " << ops.size();
    // operator
    for (int i = 0; i < ops.size(); ++i) {
      LOG(WARNING) << "name: "<<ops.at(i)->name << "\t"
      << "type: "<<ops.at(i)->type << "\t"
      << "inputs size: "<<ops.at(i)->inputs.size() << "\t"
      << "outputs size: "<<ops.at(i)->outputs.size() << "\t";
    }

    // operand,input and linear
    for (int i = 0; i < 2; ++i) {
      LOG(WARNING) << "operator name: "<<ops.at(i)->name << "\t";
      // input
       for(int j = 0; j < ops.at(i)->inputs.size(); ++j) {
          auto operand = ops.at(i)->inputs.at(j);
          LOG(WARNING) << "input operand name: "<<operand->name << "\t"
          << "operand type: "<<operand->type << "\t"
          << "operand shape: "<<operand->shape.size() << "\t";
      }
      // output
      for(int j = 0; j < ops.at(i)->outputs.size(); ++j) {
          auto operand = ops.at(i)->outputs.at(j);
          LOG(WARNING) << "output operand name: "<<operand->name << "\t"
          << "operand type: "<<operand->type << "\t"
          << "operand shape: "<<operand->shape.size() << "\t";
      }
    }
}

  int main(int argc, char** argv) {
    vega_rt::init_logging(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}