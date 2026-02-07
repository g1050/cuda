#pragma once
#include "dtype.hpp"
#include <memory>
#include <string>
#include <vector>
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

	explicit Parameter(ParameterType type = ParameterType::Unknown)
		: type_(type) {}
	ParameterType type_ = ParameterType::Unknown;
	};

	struct ParameterInt : public Parameter {
	ParameterInt() : Parameter(ParameterType::Int) {}
	int value_ = 0;
	};

	struct ParameterFloat : public Parameter {
	ParameterFloat() : Parameter(ParameterType::Float) {}
	float value_ = 0.f;
	};

	struct ParameterString : public Parameter {
	ParameterString() : Parameter(ParameterType::String) {}
	std::string value_;
	};

	struct ParameterIntArray : public Parameter {
	ParameterIntArray() : Parameter(ParameterType::IntArray) {}
	std::vector<int> value_;
	};

	struct ParameterFloatArray : public Parameter {
	ParameterFloatArray() : Parameter(ParameterType::FloatArray) {}
	std::vector<float> value_;
	};

	struct ParameterStringArray : public Parameter {
	ParameterStringArray() : Parameter(ParameterType::StringArray) {}
	std::vector<std::string> value_;
	};

	struct ParameterBool : public Parameter {
	ParameterBool() : Parameter(ParameterType::Bool) {}
	bool value_ = false;
	};
	using ParameterSP = std::shared_ptr<Parameter>;
} // namespace vega_rt