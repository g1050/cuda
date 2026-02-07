#pragma once
#include "dtype.hpp"
#include <cstdint>
#include <vector>
namespace vega_rt {
    /**
     * @brief 权重信息，用一个char数组保存
     * 
     */
    struct Attribute {
        public:
            virtual ~Attribute();
            std::vector<char> weight_data;
            std::vector<uint32_t> shape ; 
    };
}