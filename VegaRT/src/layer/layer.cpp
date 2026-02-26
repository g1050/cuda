#include "layer/layer.hpp"
#include "dtype.hpp"
#include "operator.hpp"
#include "operand.hpp"
#include "tensor.hpp"
#include <glog/logging.h>
#include "utils.hpp"
namespace vega_rt {
    VegaError Layer::Forward() {
        const auto &operator_sp = operator_.lock();
        const std::vector<OperandSP> &input_operands = operator_sp->input_operands_list_;

        // 输入Tensors
        std::vector<TensorSP> input_tensors;
        for(const auto &input_operand : input_operands) {
            for(const auto &input_tensor : input_operand->datas_) { // 按batch
                input_tensors.push_back(input_tensor);
            }
        }
        CHECK(!input_tensors.empty())
        << operator_sp->name_ << " Layer input data is empty";

        // Check输出Tensors
        CHECK(operator_sp->output_operand_ != nullptr && !operator_sp->output_operand_->datas_.empty())
        << operator_sp->name_ << " Layer output data is empty";
        return operator_sp->layer_->Forward(input_tensors, operator_sp->output_operand_->datas_);

        return VegaError::Success;
    }

    VegaError Layer::Forward(const std::vector<TensorSP> &inputs, std::vector<TensorSP> &outputs) {
        
        return VegaError::Success;
    }

    ParameterLayer::ParameterLayer(std::string layer_name) : Layer(layer_name) {}

    void ParameterLayer::InitWeightParameters(const uint32_t param_count, const uint32_t param_channel,
        const uint32_t param_height, const uint32_t param_width) {
        this->weight_tensors_ = std::vector<TensorSP>(param_count);
        for(uint32_t i = 0; i < param_count; ++i) {
            this->weight_tensors_.at(i) = vega_rt::TensorCreate(param_channel, param_height, param_width);
        }
    }

    void ParameterLayer::InitBiasParameters(const uint32_t param_count, const uint32_t param_channel,
        const uint32_t param_height, const uint32_t param_width) {
        this->bias_tensors_ = std::vector<TensorSP>(param_count);
        for(uint32_t i = 0; i < param_count; ++i) {
            this->bias_tensors_.at(i) = vega_rt::TensorCreate(param_channel, param_height, param_width);
        }
    }

    void ParameterLayer::set_weights(const std::vector<TensorSP> &weights) {
        CHECK(weights.size() == this->weight_tensors_.size())
        << "Weight size mismatch";
        for(uint32_t i = 0; i < weights.size(); ++i) {
            CHECK(weights.at(i) != nullptr)
            << "Weight is nullptr";
            CHECK(weights.at(i)->shape() == this->weight_tensors_.at(i)->shape())
            << "Weight shape mismatch";
        }
        this->weight_tensors_ = weights;
    }
}