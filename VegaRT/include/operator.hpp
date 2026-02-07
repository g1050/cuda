#pragma once
#include "tensor.hpp"
#include "parameter.hpp"
#include "attribute.hpp"
namespace vega_rt {

    /**
     * @brief 计算节点
     * 
     */
    struct Operator {
        public:
            virtual ~Operator();

            // 节点属性
            std::string name;
            std::string type;
            // 输入,输出
            std::vector<std::string> input_names;
            std::vector<std::string> output_names;

            // 参数信息
            std::map<std::string, Parameter> params;
            // 属性信息
            std::map<std::string, Attribute> attrs;
    };

}