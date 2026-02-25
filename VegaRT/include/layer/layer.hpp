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
        public:
            explicit ParameterLayer(std::string layer_name);

            /**
             * @brief 初始化权重
             * 
             * @param param_count 卷积核的数量
             * @param param_channel 卷积核的通道数
             * @param param_height 卷积核的高度
             * @param param_width 卷积核的宽度
             */
            void InitWeightParameters(const uint32_t param_count, const uint32_t param_channel,
                const uint32_t param_height, const uint32_t param_width);

            /**
             * @brief 初始化偏置
             * 
             * @param param_count 偏置的数量
             * @param param_channel 偏置的通道数
             * @param param_height 偏置的高度
             * @param param_width 偏置的宽度
             */
            void InitBiasParameters(const uint32_t param_count, const uint32_t param_channel,
                const uint32_t param_height, const uint32_t param_width);

        protected:
            std::vector<TensorSP> weight_tensors_;
            std::vector<TensorSP> bias_tensors_;
    };
}