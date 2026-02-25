#include "layer/vision/pooling.hpp"
#include <memory>
namespace vega_rt {
    MaxPoolingLayer::MaxPoolingLayer(uint32_t padding_h, uint32_t padding_w, 
                                    uint32_t pooling_size_h, uint32_t pooling_size_w,
                                    uint32_t stride_h, uint32_t stride_w) : NonParameterLayer("nn.MaxPool2d"),
        padding_h_(padding_h), padding_w_(padding_w),
        pooling_size_h_(pooling_size_h), pooling_size_w_(pooling_size_w),
        stride_h_(stride_h), stride_w_(stride_w) {
    }

    VegaError MaxPoolingLayer::Forward(const std::vector<TensorSP> &inputs, std::vector<TensorSP> &outputs) {
        if(inputs.empty()) {
            LOG(ERROR) << "inputs is empty";
            return VegaError::InvalidParam;
        }
        if(inputs.size() != outputs.size()) {
            LOG(ERROR) << "inputs and outputs size mismatch";
            return VegaError::InvalidParam;
        }
        const uint32_t batch_size = inputs.size();
        if (stride_h_ <= 0 || stride_w_<= 0) {
            LOG(ERROR) << "stride_h and stride_w must be greater than 0";
            return VegaError::InvalidParam;
        }
        for(uint32_t i = 0; i < batch_size; i++) {
            const auto &input = inputs.at(i);
            auto &output = outputs.at(i);
            if(input == nullptr || input->empty()) {
                LOG(ERROR) << "input is nullptr or empty";
                return VegaError::InvalidParam;
            }

            uint32_t input_h = input->rows();
            uint32_t input_w = input->cols();
            uint32_t output_h = (input_h + 2 * padding_h_ - pooling_size_h_) / stride_h_ + 1;
            uint32_t output_w = (input_w + 2 * padding_w_ - pooling_size_w_) / stride_w_ + 1;
            output = vega_rt::TensorCreate(input->channels(), output_h, output_w);
            for(uint32_t c = 0; c < input->channels(); c++) {
                const auto &input_slice = input->slice(c);
                auto &output_slice = output->slice(c);

                // 遍历输出特征图上的每一个位置 (oh, ow)
                for(uint32_t oh = 0; oh < output_h; ++oh) {
                    for(uint32_t ow = 0; ow < output_w; ++ow) {
                        float max_val = -FLT_MAX;

                        // 计算以 (oh, ow) 为中心的池化窗口在输入上的覆盖区域
                        for(uint32_t kh = 0; kh < pooling_size_h_; ++kh) {
                            for(uint32_t kw = 0; kw < pooling_size_w_; ++kw) {
                                // 映射到原始输入坐标（考虑 stride 和 padding）
                                int in_row = static_cast<int>(oh * stride_h_) +
                                             static_cast<int>(kh) - static_cast<int>(padding_h_);
                                int in_col = static_cast<int>(ow * stride_w_) +
                                             static_cast<int>(kw) - static_cast<int>(padding_w_);

                                // 只有落在有效输入区域内的才参与 max
                                if (in_row >= 0 && in_row < static_cast<int>(input_h) &&
                                    in_col >= 0 && in_col < static_cast<int>(input_w)) {
                                    float val = input_slice(in_row, in_col);
                                    max_val = std::max(max_val, val);
                                }
                            }
                        }

                        output_slice(oh, ow) = max_val;
                    }
                }
            }
        }
        return VegaError::Success;
    }

    VegaError MaxPoolingLayer::GetInstance(const std::shared_ptr<Operator>& op, std::shared_ptr<Layer>& pooling_layer) {
        CHECK(op != nullptr) << "MaxPooling operator is nullptr";
        // 根据op的参数来查找maxpooling对应的参数
        auto stride_param = op->params_.find("stride");
        auto kernel_param = op->params_.find("kernel_size");
        auto padding_param = op->params_.find("padding");
        if(stride_param == op->params_.end() || kernel_param == op->params_.end() || padding_param == op->params_.end()) {
            LOG(ERROR) << "MaxPooling operator params is not found";
            return VegaError::InvalidParam;
        }
        std::vector<int> stride = std::dynamic_pointer_cast<ParameterIntArray>(stride_param->second)->value_;
        std::vector<int> kernel = std::dynamic_pointer_cast<ParameterIntArray>(kernel_param->second)->value_;
        std::vector<int> padding = std::dynamic_pointer_cast<ParameterIntArray>(padding_param->second)->value_;
        // 参数校验,maxpool2d
        if(stride.size() != 2 || kernel.size() != 2 || padding.size() != 2) {
            LOG(ERROR) << "MaxPooling operator params is not valid";
            return VegaError::InvalidParam;
        }
        pooling_layer = std::make_shared<MaxPoolingLayer>(padding[0], padding[1], kernel[0], kernel[1], stride[0], stride[1]);
        return VegaError::Success;
    }
    LayerRegistererWrapper kMaxPoolingGetInstance("nn.MaxPool2d", MaxPoolingLayer::GetInstance);
}