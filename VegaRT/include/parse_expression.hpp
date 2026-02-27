#pragma once
#include "dtype.hpp"
#include <cstdint>
#include <string>
#include <vector>
#include <glog/logging.h>
namespace vega_rt {
    
    /*词元类型*/
    enum class TokenType{
        TokenUnknown = -9, // 未知词元
        TokenInputNumber = -8, // 输入编号
        TokenComma = -7, // 逗号
        TokenAdd = -6, // 加号
        TokenMul = -5, // 乘号
        TokenLeftBracket = -4, // 左括号
        TokenRightBracket = -3, // 右括号
        TokenSin = -2, // 正弦函数
    };

     struct Token{
        TokenType type_; //词元类型
        int32_t start_pos; //表达式中词元起始位置
        int32_t end_pos; //表达式中词元结束位置
        Token(TokenType type, int32_t start_pos, int32_t end_pos) : type_(type), start_pos(start_pos), end_pos(end_pos) {}
     };

     /**表达式解析器，将表达式字符串分割成词元*/
     class ExpressionParser{
        public:
            explicit ExpressionParser(const std::string& expression);
            ~ExpressionParser();
            VegaError Tokenize(bool retokenize = false);
            /**获取分割后的词元结构 */
            const std::vector<Token>& Tokens() const;
            /**获取分割后的表达式字符串 */
            const std::vector<std::string>& TokensStr() const;
        private:
            std::string expression_;
            std::vector<Token> tokens_;
            std::vector<std::string> tokens_str_; // 被分割的表达式字符串
            
     };
    
}