#pragma once
#include <string>
#include <vector>
#include "dtype.hpp"
#include "tensor.hpp"
namespace vega_rt {
    /**
     * @brief 将形状转换为字符串
     * 
     * @param shapes 形状
     * @return std::string 形状字符串
     */
    std::string ShapeStr(const std::vector<int> &shapes);
    /**
     * @brief 将参数类型转换为字符串
     * 
     * @param type 参数类型
     * @return std::string 参数类型字符串
     */
    std::string ParameterTypeStr(ParameterType type);
    /**
     * @brief 将数据类型转换为字符串
     *
     * @param type 数据类型
     * @return std::string 数据类型字符串
     */
    std::string DataTypeStr(DataType type);
    TensorSP TensorCreate(uint32_t channels, uint32_t rows,
        uint32_t cols);
    TensorSP TensorCreate(uint32_t rows, uint32_t cols);
    TensorSP TensorCreate(uint32_t size);
    TensorSP TensorCreate(const std::vector<uint32_t> &shapes);
}