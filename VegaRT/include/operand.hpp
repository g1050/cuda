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
            std::string name;
            DataType dtype = DataType::Float32; // 默认fp32
            std::vector<uint32_t> shapes;
            std::vector<TensorSP> datas;
    };
}