#include "dtype.hpp"
#include <parse_expression.hpp>

namespace vega_rt {

    ExpressionParser::ExpressionParser(const std::string& expression) : expression_(expression) {}
    ExpressionParser::~ExpressionParser() {}
    
    VegaError ExpressionParser::Tokenize(bool retokenize) {
        LOG(ERROR) << "ExpressionParser::Tokenize not implemented";
        return VegaError::Success;
    }

    const std::vector<Token>& ExpressionParser::Tokens() const {
        return tokens_;
    }

    const std::vector<std::string>& ExpressionParser::TokensStr() const {
        return tokens_str_;
    }


}