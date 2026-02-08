#include "operator.hpp"
#include "common.hpp"
#include <glog/logging.h>
#include "utils.hpp"
namespace vega_rt {
    
    Operator::~Operator() {
        //析构Operator
    }

    VegaError OperatorUtils::InitOperatorInput(std::vector<OperatorSP> operator_spv) {
        if (operator_spv.empty()) {
            LOG(ERROR) << "operator_spv is empty";
            return VegaError::InvalidParam;
        }
        for(const auto &operator_sp : operator_spv) {
            if (operator_sp->input_operands_list_.empty()) { // 输入节点没有输入操作数，跳过
                continue;
            }
            for(const auto &[input_name, input_operand] : operator_sp->input_operands_map_) {
                CHECK(input_operand->dtype_ == DataType::Float32) << "only support float32";
                auto shape = input_operand->shapes_;
                auto datas = input_operand->datas_;
                auto batch_size = shape[0];
                CHECK_NE(shape.size(),0) << "shape is empty";
                //  不支持动态batch
                CHECK_GT(batch_size, 0) << "batch size must be greater than 0";
                // 只支持2D,4D,3D
                CHECK(shape.size() == 2 ||
                shape.size() == 4 ||
                shape.size() == 3)
                    << "Unsupported tensor shape sizes: " << shape.size();
                if(datas.empty()) {
                    datas.resize(batch_size); // 初始化batch_size个空tensor
                }else{
                    CHECK_EQ(datas.size(), batch_size) << "data size must be equal to batch size";
                }
            }
        }
        return VegaError::Success;
    }

    VegaError OperatorUtils::InitOperatorOutput(const std::vector<pnnx::Operator *> &pnnx_operators,
                const std::vector<OperatorSP> &operators) {
        if (pnnx_operators.empty() || operators.empty()) {
            LOG(ERROR) << "pnnx_operators or operators is empty";
            return VegaError::InvalidParam;
        }
        for(auto i = 0; i < pnnx_operators.size(); i++) {
            // 获取pnnx算子的操作数
            const std::vector<pnnx::Operand*> &pnnx_operands = pnnx_operators.at(i)->outputs;
            if(pnnx_operands.empty()) continue;
            CHECK_EQ(pnnx_operands.size(), 1) << "only support single output"; // 只支持单输出的算子
            
            const pnnx::Operand *operand = pnnx_operands.front();
            CHECK(operand != nullptr) << "Operand output is null";
            const std::vector<int32_t> &operand_shapes = operand->shape;
            CHECK_NE(operand_shapes.size(), 0) << "shape is empty";
            auto batch_size = operand_shapes.at(0);
            CHECK_GT(batch_size, 0) << "batch size must be greater than 0";
            // 只支持2D,4D,3D
            CHECK(operand_shapes.size() == 2 ||
            operand_shapes.size() == 4 ||
            operand_shapes.size() == 3)
                << "Unsupported tensor shape sizes: " << operand_shapes.size();


            const auto &vega_operator = operators.at(i);
            auto &output_operand = vega_operator->output_operand_;

            if(output_operand == nullptr) { // 第一次初始化，底层创建了arma::fcube
                output_operand = std::make_shared<Operand>();
                for(auto j = 0;j<batch_size;j++) {
                    if(operand_shapes.size() == 2) { //不包括batch_size
                        output_operand->datas_.push_back(TensorCreate(operand_shapes.at(1)));
                    }else if(operand_shapes.size() == 3) {
                        output_operand->datas_.push_back(TensorCreate(operand_shapes.at(1), operand_shapes.at(2)));
                    }else if(operand_shapes.size() == 4) {
                        output_operand->datas_.push_back(TensorCreate(operand_shapes.at(1), operand_shapes.at(2), operand_shapes.at(3)));
                    }
                }
            }else{ //已经初始化过，校验shape
                CHECK(batch_size == output_operand->datas_.size());
                CHECK(output_operand->dtype_ == DataType::Float32);
                CHECK(output_operand->shapes_ == operand_shapes);
                for(auto j = 0;j<batch_size;j++) {
                    // current batch
                    auto current_batch_tensor = output_operand->datas_.at(j);
                    const auto tensor_shape = current_batch_tensor->shape();
                    auto operand_shape = operand_shapes; //去掉batch_size
                    operand_shape.erase(operand_shape.begin());
                    if(tensor_shape != std::vector<uint32_t>(operand_shape.begin(), operand_shape.end())) {
                        // todo: resize
                        CHECK(false) << "tensor shape must be equal to operand shape";
                    }
                }
            }


        }
        return VegaError::Success;
    }

}