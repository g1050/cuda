#pragma once
#include "tensor.hpp"
#include "parameter.hpp"
#include "attribute.hpp"
#include "operand.hpp"
namespace vega_rt {
    class Operator;
    using OperatorSP = std::shared_ptr<Operator>;

    /**
     * @brief 计算节点
     * 
     */
    struct Operator {
        public:
            virtual ~Operator();

            // 节点属性
            std::string name_;
            std::string type_;

            // 参数信息
            std::map<std::string, ParameterSP> params_;
            // 权重信息
            std::map<std::string, AttributeSP> attrs_;
            
            // 输入操作数
            std::map<std::string, OperandSP> input_operands_map_;
            std::vector<OperandSP> input_operands_list_;

            // 输出操作数
            std::vector<std::string> output_names_;

            // todo: 计算节点Layer

    };
}