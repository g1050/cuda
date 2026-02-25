#include "layer/vision/conv.hpp"
namespace vega_rt {
    ConvLayer::ConvLayer(uint32_t output_channels, uint32_t input_channels, 
        uint32_t kernel_h, uint32_t kernel_w, 
        uint32_t padding_h, uint32_t padding_w,
        uint32_t stride_h, uint32_t stride_w,
        uint32_t groups,bool use_bias) : ParameterLayer("nn.Conv2d"),
        output_channels_(output_channels), input_channels_(input_channels),
        kernel_h_(kernel_h), kernel_w_(kernel_w),
        padding_h_(padding_h), padding_w_(padding_w),
        stride_h_(stride_h), stride_w_(stride_w),
        groups_(groups), use_bias_(use_bias) {
            // 分组卷积
            if(groups_ != 1){
                input_channels_ /= groups_;
            }
            this->InitWeightParameters(output_channels_, input_channels_, kernel_h_, kernel_w_);
            if(use_bias_){
                this->InitBiasParameters(output_channels_, 1, 1, 1);
            }
    }

    VegaError ConvLayer::Forward(const std::vector<TensorSP> &inputs, std::vector<TensorSP> &outputs) {
        LOG(ERROR) << "ConvLayer::Forward not implemented";
        return VegaError::InvalidParam;
    }

    VegaError ConvLayer::GetInstance(const std::shared_ptr<Operator>& op, std::shared_ptr<Layer>& conv_layer) {
        CHECK(op != nullptr) << "Conv operator is nullptr";
        conv_layer = std::make_shared<ConvLayer>(1,1,1,1,1,1,1,1,1,true);
        return VegaError::Success;
    }
    LayerRegistererWrapper kConvGetInstance("nn.Conv2d", ConvLayer::GetInstance);

}