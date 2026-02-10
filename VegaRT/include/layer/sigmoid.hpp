#pragma once
#include "layer.hpp"
namespace vega_rt {
    class SigmoidLayer : public NonParameterLayer {
        public:
            SigmoidLayer() : NonParameterLayer("nn.Sigmoid") {}
            SigmoidLayer(std::string layer_name) : NonParameterLayer(layer_name) {}
            VegaError Forward(const std::vector<TensorSP> &inputs, std::vector<TensorSP> &outputs) override;
            static VegaError GetInstance(
                const std::shared_ptr<Operator>& op,
                std::shared_ptr<Layer>& sigmoid_layer);
    };
    using SigmoidLayerSP = std::shared_ptr<SigmoidLayer>;
}