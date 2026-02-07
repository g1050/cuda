#pragma once
#include <cstdint>
#include <memory>
#include <vector>
#include "dtype.hpp"
namespace vega_rt {
    /**
     * @brief 权重信息，用一个char数组保存
     * 
     */
    struct Attribute {
        public:
            virtual ~Attribute();
            std::vector<char> weight_data_;
            std::vector<int> shape_ ; 
            DataType type_ = DataType::Unknown; // 只支持fp32
    };
    using AttributeSP = std::shared_ptr<Attribute>;
}