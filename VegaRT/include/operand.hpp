#pragma once
#include "dtype.hpp"
#include "tensor.hpp"
#include <vector>
#include <memory>
namespace vega_rt {
    /**
     * @brief 操作数
     * 
     */
    struct Operand {
        public:
            virtual ~Operand();
            std::string name_;
            DataType dtype_ = DataType::Float32; // 默认fp32
            std::vector<int> shapes_; //为了和pnnx保持一致，使用了int类型
            std::vector<TensorSP> datas_; // 具体存放数据结构
    };
    using OperandSP = std::shared_ptr<Operand>;
}