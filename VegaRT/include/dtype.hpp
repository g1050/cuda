#pragma once

namespace vega_rt {

    enum class DataType {
        Unknown = 0,
        Float32 = 1,
        Float64 = 2,
        Float16 = 3,
        Int32 = 4,
        Int64 = 5,
        Int16 = 6,
        Int8 = 7,
        UInt8 = 8,
    };

    enum class ParameterType {
        Unknown = 0,
        Bool = 1,
        Int = 2,
        Float = 3,
        String = 4,
        IntArray = 5,
        FloatArray = 6,
        StringArray = 7,
      };
}