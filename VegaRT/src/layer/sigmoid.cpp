#include "layer/sigmoid.hpp"
#include "layer/layer_factory.hpp"
#include <glog/logging.h>
#include "utils.hpp"
namespace vega_rt {
    VegaError SigmoidLayer::Forward(const std::vector<TensorSP> &inputs, std::vector<TensorSP> &outputs) {
        //check 
        if(inputs.empty()) {
            LOG(ERROR) << "inputs is empty";
            return VegaError::InvalidParam;
        }
        if(inputs.size() != outputs.size()) {
            LOG(ERROR) << "inputs and outputs size mismatch";
            return VegaError::InvalidParam;
        }

        const uint32_t batch_size = inputs.size();
        for(uint32_t i = 0; i < batch_size; i++) {
            const auto &input = inputs.at(i);
            const auto &output = outputs.at(i);
            if(input == nullptr || input->empty()) {
                LOG(ERROR) << "input is nullptr or empty";
                return VegaError::InvalidParam;
            }

            if(output != nullptr && !output->empty()) 
                CHECK(input->shape() == output->shape())
                    << "input and output shape mismatch";
        }

        for (uint32_t i = 0; i < batch_size; ++i) {
            const TensorSP &input = inputs.at(i);
            CHECK(input == nullptr || !input->empty())
                    << "The input tensor array in the sigmoid layer has an empty tensor " << i
                    << " th";
        
            TensorSP output = outputs.at(i);
            // 如果没有配置输出空间,内部创建
            if (output == nullptr || output->empty()) {
              LOG(WARNING)
                  << "The output tensor array in the sigmoid layer has an empty tensor "
                  << i << " th";
              output = std::make_shared<Tensor<float>>(input->shapes());
              outputs.at(i) = output;
            }
            CHECK(output->shapes() == input->shapes())
                    << "The input and output tensor shapes of the sigmoid layer do not match "
                    << i << " th";
            // for (uint32_t j = 0; j < input->size(); ++j) {
            //   float value = input->data().at(j);
            //   output->data().at(j) = value > 0.f ? value : 0.f;
            // }
            output->data() = 1.f / (1.f + arma::exp(-input->data()));
          }
        return VegaError::Success;
    }

    VegaError SigmoidLayer::GetInstance(const std::shared_ptr<Operator>& op, std::shared_ptr<Layer>& sigmoid_layer) {
        CHECK(op != nullptr) << "sigmoid operator is nullptr";
        sigmoid_layer = std::make_shared<SigmoidLayer>();
        return VegaError::Success;
    }
    LayerRegistererWrapper kSigmoidGetInstance("nn.Sigmoid", SigmoidLayer::GetInstance);

}