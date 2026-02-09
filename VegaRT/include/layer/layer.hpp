#pragma once
#include "tensor.hpp"
#include <string>
#include <vector>
#include <memory>
#include "../dtype.hpp"
namespace vega_rt {
    class Operator;
    class Layer {
        public:
            virtual ~Layer() = default;
            explicit Layer(std::string layer_name) : layer_name_(std::move(layer_name)) {}

            /**
             * @brief 前向传播,继承类需要实现
             * 
             * @param inputs 
             * @param outputs 
             * @return VegaError 
             */
            virtual VegaError Forward(const std::vector<TensorSP> &inputs, std::vector<TensorSP> &outputs);

            /**
             * @brief 准备输入和输出数据，并且调用派生类的Forward方法
             * 
             * @return VegaError 
             */
            virtual VegaError Forward();

        protected:
            std::string layer_name_;
            std::weak_ptr<Operator> operator_; // 弱引用，用于从opearator中获取操作数、权重
    };
    using LayerSP = std::shared_ptr<Layer>;
    using NonParameterLayer = Layer;
    class ParameterLayer : public Layer {
    };
}