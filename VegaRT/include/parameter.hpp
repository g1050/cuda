#pragma once
#include "dtype.hpp"
namespace vega_rt {
/**
 * 计算节点中的参数信息，参数一共可以分为如下的几类
 * 1.int
 * 2.float
 * 3.string
 * 4.bool
 * 5.int array
 * 6.string array
 * 7.float array
 */

 /**
  * @brief 参数类型基类，不是基类保存所有类型的属性，而是根据需要派生
  * 
  */
 struct Parameter { /// 计算节点中的参数信息
    virtual ~Parameter() = default;
  
    explicit Parameter(ParameterType type = ParameterType::Unknown) : type(type) {
  
    }
    ParameterType type = ParameterType::Unknown;
  };
  
  struct RuntimeParameterInt : public Parameter {
    RuntimeParameterInt() : Parameter(ParameterType::Int) {
  
    }
    int value = 0;
  };
  
  struct RuntimeParameterFloat : public Parameter {
    RuntimeParameterFloat() : Parameter(ParameterType::Float) {
  
    }
    float value = 0.f;
  };
  
  struct RuntimeParameterString : public Parameter {
    RuntimeParameterString() : Parameter(ParameterType::String) {
  
    }
    std::string value;
  };
  
  struct RuntimeParameterIntArray : public Parameter {
    RuntimeParameterIntArray() : Parameter(ParameterType::IntArray) {
  
    }
    std::vector<int> value;
  };
  
  struct RuntimeParameterFloatArray : public Parameter {
    RuntimeParameterFloatArray() : Parameter(ParameterType::FloatArray) {
  
    }
    std::vector<float> value;
  };
  
  struct RuntimeParameterStringArray : public Parameter {
    RuntimeParameterStringArray() : Parameter(ParameterType::StringArray) {
  
    }
    std::vector<std::string> value;
  };
  
  struct RuntimeParameterBool : public Parameter {
    RuntimeParameterBool() : Parameter(ParameterType::Bool) {
  
    }
    bool value = false;
  };
  }