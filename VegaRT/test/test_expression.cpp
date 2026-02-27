#include "common.hpp"
#include <gtest/gtest.h>
#include <iostream>
#include "parse_expression.hpp"
using namespace vega_rt;
TEST(test_expression, test_expression_parser) {
    std::string expression = "add(@0,mul(@1,@2))";
    vega_rt::ExpressionParser parser(expression);
    parser.Tokenize();
    const auto& tokens = parser.Tokens();
    const auto& tokens_str = parser.TokensStr();
    for(const auto& token_str : tokens_str) {
        LOG(INFO) << "token_str: " << token_str;
    }
}

int main(int argc, char** argv) {
    vega_rt::init_logging(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}