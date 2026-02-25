#pragma once
#include "../layer.hpp"
#include <cstdint>
#include "../layer_factory.hpp"
#include "../../utils.hpp"
namespace vega_rt {
    class MaxPoolingLayer : public NonParameterLayer {
        public:
            explicit MaxPoolingLayer(uint32_t padding_h, uint32_t padding_w, 
                                    uint32_t pooling_size_h, uint32_t pooling_size_w,
                                    uint32_t stride_h, uint32_t stride_w);
            
            VegaError Forward(const std::vector<TensorSP> &inputs, std::vector<TensorSP> &outputs) override;
            static VegaError GetInstance(
                const std::shared_ptr<Operator>& op,
                std::shared_ptr<Layer>& pooling_layer);
        private:
            uint32_t padding_h_ = 0;
            uint32_t padding_w_ = 0;
            uint32_t pooling_size_h_ = 0;
            uint32_t pooling_size_w_ = 0;
            uint32_t stride_h_ = 1;
            uint32_t stride_w_ = 1;
    };
}