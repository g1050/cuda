#pragma once
#include "layer.hpp"
namespace vega_rt {
    class ReluLayer : public NonParameterLayer {
        public:
            ReluLayer() : NonParameterLayer("nn.ReLU") {}
            ReluLayer(std::string layer_name) : NonParameterLayer(layer_name) {}
            VegaError Forward(const std::vector<TensorSP> &inputs, std::vector<TensorSP> &outputs) override;
            static VegaError GetInstance(
                const std::shared_ptr<Operator>& op,
                std::shared_ptr<Layer>& relu_layer);
    };
    using ReluLayerSP = std::shared_ptr<ReluLayer>;
}