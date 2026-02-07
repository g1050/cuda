#include "tensor.hpp"
#include "common.hpp"
#include "operator.hpp"
#include <gtest/gtest.h>
#include <iostream>

TEST(OperatorTest, TestOperatorInit) {
    vega_rt::Operator operator1;
    operator1.name = "add";
    operator1.type = "add";
    operator1.input_names = {"input1", "input2"};
    operator1.output_names = {"output"};
}


int main(int argc, char** argv) {
    vega_rt::init_logging(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}