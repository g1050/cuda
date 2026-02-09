#pragma once
#include "tensor.hpp"
#include "parameter.hpp"
#include "attribute.hpp"
#include "operand.hpp"
#include <pnnx/ir.h>
#include "layer/layer.hpp"
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
            OperandSP output_operand_; //只支持单输出的算子

            // 输出节点
            std::map<std::string, OperatorSP> output_operators_map_;

            // 计算节点Layer
            std::shared_ptr<Layer> layer_;

            // 标记该节点是否已读，用于递归拓扑排序
            bool has_forward_ = false;

    };

    class OperatorUtils {
        public:
            /**
             * @brief 初始化算子的输入
             * 
             * @param operator_spv 
             * @return VegaError 
             */
            static VegaError InitOperatorInput(std::vector<OperatorSP> operator_spv);
            /**
             * @brief 初始化算子的输出
             * 
             * @param pnnx_operators pnnx算子
             * @param operators vega算子
             * @return VegaError 
             */
            static VegaError InitOperatorOutput(const std::vector<pnnx::Operator *> &pnnx_operators,
                const std::vector<OperatorSP> &operators); 
    };
}