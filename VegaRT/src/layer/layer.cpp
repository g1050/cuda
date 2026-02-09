#include "layer/layer.hpp"
#include "operator.hpp"
#include "operand.hpp"
#include <glog/logging.h>

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
    
}