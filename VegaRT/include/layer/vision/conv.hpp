#pragma once
#include "../layer.hpp"
#include <cstdint>
#include "../layer_factory.hpp"
namespace vega_rt {
    class ConvLayer : public ParameterLayer {
        public:
            explicit ConvLayer(uint32_t output_channels, uint32_t input_channels, 
                uint32_t kernel_h, uint32_t kernel_w, 
                uint32_t padding_h, uint32_t padding_w,
                uint32_t stride_h, uint32_t stride_w,
                uint32_t groups,bool use_bias = true);
            
            VegaError Forward(const std::vector<TensorSP> &inputs, std::vector<TensorSP> &outputs) override;
            static VegaError GetInstance(
                const std::shared_ptr<Operator>& op,
                std::shared_ptr<Layer>& conv_layer);
        private:
            uint32_t output_channels_;
            uint32_t input_channels_;
            uint32_t kernel_h_;
            uint32_t kernel_w_;
            uint32_t padding_h_;
            uint32_t padding_w_;
            uint32_t stride_h_;
            uint32_t stride_w_;
            uint32_t groups_;
            bool use_bias_;
    };
    using ConvLayerSP = std::shared_ptr<ConvLayer>;
}