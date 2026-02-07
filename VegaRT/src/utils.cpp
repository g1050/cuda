#include "utils.hpp"
#include "dtype.hpp"
#include <sstream>

namespace vega_rt {
    std::string ShapeStr(const std::vector<int> &shapes) {
        std::ostringstream ss;
        for (int i = 0; i < shapes.size(); ++i) {
            ss << shapes.at(i);
            if (i != shapes.size() - 1) {
                ss << " x ";
            }
        }
        return ss.str();
    }
    std::string ParameterTypeStr(ParameterType type) {
        switch (type) {
            case ParameterType::Bool:
                return "bool";
            case ParameterType::Int:
                return "int";
            case ParameterType::Float:
                return "float";
            case ParameterType::String:
                return "string";
            case ParameterType::IntArray:
                return "int array";
            case ParameterType::FloatArray:
                return "float array";
            case ParameterType::StringArray:
                return "string array";
            case ParameterType::Unknown:
                return "unknown";
            default:
                return "unknown";
        }
    }

    std::string DataTypeStr(DataType type) {
        switch (type) {
            case DataType::Float32:
                return "float32";
            case DataType::Float64:
                return "float64";
            case DataType::Float16:
                return "float16";
            case DataType::Int32:
                return "int32";
            case DataType::Int64:
                return "int64";
            case DataType::Int16:
                return "int16";
            case DataType::Int8:
                return "int8";
            case DataType::UInt8:
                return "uint8";
            case DataType::Unknown:
                return "unknown";
            default:
                return "unknown";
        }
    }
}