#include "graph.hpp"
#include "dtype.hpp"
#include "attribute.hpp"

namespace vega_rt {

    Graph::Graph(const std::string& param_path, const std::string& bin_path) {
        this->param_path_ = param_path;
        this->bin_path_ = bin_path;

    }

    Graph::~Graph() {}

    const std::vector<OperatorSP>& Graph::operators() const {
        return this->operators_list_;
    }

    VegaError Graph::Init() {
        // check path valid
        if (param_path_.empty() || bin_path_.empty()) {
            LOG(ERROR) << "param_path or bin_path is empty";
            return VegaError::InvalidParam;
        }

        pnnx_graph_ = std::make_unique<pnnx::Graph>();
        // 创建指针的同时实例化对象
        int load_result = pnnx_graph_->load(param_path_, bin_path_);
        if (load_result != 0) {
            LOG(ERROR) << "load pnnx graph failed";
            return VegaError::InvalidParam;
        }

        // 记录operators指针，用pnnx的即可，指针操作
        std::vector<pnnx::Operator*> pnnx_operators = pnnx_graph_->ops;
        // check
        if (pnnx_operators.empty()) {
            LOG(ERROR) << "operators is empty";
            return VegaError::InvalidParam;
        }
        // 创建 vegaGraph的operators
        this->operators_list_.clear();
        this->operators_map_.clear();
        for(const auto& pnnx_operator : pnnx_operators) {
            if (pnnx_operator == nullptr) {
                LOG(ERROR) << "pnnx_operator is nullptr";
                return VegaError::InvalidParam;
            }
            // 创建vegaOperator对象
            OperatorSP operator_sp = std::make_shared<Operator>();
            operator_sp->name_ = pnnx_operator->name;
            operator_sp->type_ = pnnx_operator->type;

            // 输入操作数初始化
            InitGraphInputOperands(operator_sp, pnnx_operator->inputs);
            // 输出
            InitGraphOutputOperands(operator_sp, pnnx_operator->outputs);
            
            // 初始化参数
            std::map<std::string, pnnx::Parameter> params = pnnx_operator->params;
            InitGraphParams(operator_sp, params);
            // 初始化权重
            InitGraphAttrs(operator_sp, pnnx_operator->attrs);
            this->operators_list_.push_back(operator_sp);
            this->operators_map_.insert(std::make_pair(operator_sp->name_, operator_sp));
        }


        return VegaError::Success;
    }

    VegaError Graph::InitGraphInputOperands(OperatorSP operator_sp, std::vector<pnnx::Operand*> pnnx_operands) {
        for(const auto& pnnx_operand : pnnx_operands) {
            if (pnnx_operand == nullptr) {
                LOG(ERROR) << "pnnx_operand is nullptr";
                return VegaError::InvalidParam;
            }
            // vegaOperand
            OperandSP operand_sp = std::make_shared<Operand>();
            // 记录前驱节点的名字
            operand_sp->name_ = pnnx_operand->producer->name;

            // shape和type含义相同直接赋值
            operand_sp->shapes_ = pnnx_operand->shape;
            if (pnnx_operand->type != 1) {
                LOG(ERROR) << "only support float32, but pnnx_operand type is " << pnnx_operand->type;
                return VegaError::InvalidParam;
            }
            operand_sp->dtype_ = DataType::Float32;

            operator_sp->input_operands_map_.insert(std::make_pair(operand_sp->name_, operand_sp));
            operator_sp->input_operands_list_.push_back(std::move(operand_sp));
        }
        return VegaError::Success;
    }

    VegaError Graph::InitGraphOutputOperands(OperatorSP operator_sp, std::vector<pnnx::Operand*> pnnx_operands) {
        for(const auto& pnnx_operand : pnnx_operands) {
            if (pnnx_operand == nullptr) {
                LOG(ERROR) << "pnnx_operand is nullptr";
                return VegaError::InvalidParam;
            }
            for(const auto &consumer: pnnx_operand->consumers) {
                if (consumer == nullptr) {
                    LOG(ERROR) << "consumer is nullptr";
                    return VegaError::InvalidParam;
                }
                // 记录后继节点的名字
                operator_sp->output_names_.push_back(consumer->name);
            }
        }
        return VegaError::Success;
    }

    VegaError Graph::InitGraphParams(OperatorSP operator_sp, std::map<std::string, pnnx::Parameter> params) {
        // 结构化绑定
        for(const auto &[name, param] : params) {
            auto type = param.type;
            switch (type) {
                case int(ParameterType::Bool):{
                    std::shared_ptr<ParameterBool> parameter_bool = std::make_shared<ParameterBool>();
                    parameter_bool->value_ = param.b;
                    operator_sp->params_.insert(std::make_pair(name, parameter_bool));
                    break;
                }
                case int(ParameterType::Int):
                {
                    std::shared_ptr<ParameterInt> parameter_int = std::make_shared<ParameterInt>();
                    parameter_int->value_ = param.i;
                    operator_sp->params_.insert(std::make_pair(name, parameter_int));
                    break;
                }
                case int(ParameterType::Float):
                {
                    std::shared_ptr<ParameterFloat> parameter_float = std::make_shared<ParameterFloat>();
                    parameter_float->value_ = param.f;
                    operator_sp->params_.insert(std::make_pair(name, parameter_float));
                    break;
                }
                case int(ParameterType::String):
                {
                    std::shared_ptr<ParameterString> parameter_string = std::make_shared<ParameterString>();
                    parameter_string->value_ = param.s;
                    operator_sp->params_.insert(std::make_pair(name, parameter_string));
                    break;
                }
                case int(ParameterType::IntArray):
                {
                    std::shared_ptr<ParameterIntArray> parameter_int_array = std::make_shared<ParameterIntArray>();
                    parameter_int_array->value_ = param.ai;
                    operator_sp->params_.insert(std::make_pair(name, parameter_int_array));
                    break;
                }
                case int(ParameterType::FloatArray):
                {
                    std::shared_ptr<ParameterFloatArray> parameter_float_array = std::make_shared<ParameterFloatArray>();
                    parameter_float_array->value_ = param.af;
                    operator_sp->params_.insert(std::make_pair(name, parameter_float_array));
                    break;
                }
                case int(ParameterType::StringArray):
                {
                    std::shared_ptr<ParameterStringArray> parameter_string_array = std::make_shared<ParameterStringArray>();
                    parameter_string_array->value_ = param.as;
                    operator_sp->params_.insert(std::make_pair(name, parameter_string_array));
                    break;
                }
                default:
                {
                    LOG(ERROR) << "unknown parameter type: " << type;
                    return VegaError::InvalidParam;
                }
            }
        }
        return VegaError::Success;
    }

    VegaError Graph::InitGraphAttrs(OperatorSP operator_sp, std::map<std::string, pnnx::Attribute> attrs) {
        for(const auto &[name, attr] : attrs) {
            auto type = attr.type;
            // 0=null 1=f32 2=f64 3=f16 4=i32 5=i64 6=i16 7=i8 8=u8 9=bool
            switch (type) {
                case 1:{
                    AttributeSP attribute_float = std::make_shared<Attribute>();
                    // 拷贝赋值
                    attribute_float->weight_data_ = attr.data;
                    attribute_float->shape_ = attr.shape;
                    attribute_float->type_ = DataType::Float32;
                    operator_sp->attrs_.insert(std::make_pair(name, attribute_float));
                    break;
                }
                default:{
                    LOG(ERROR) << "Only support float32, but pnnx_attribute type is " << type;
                    return VegaError::InvalidParam;
                }
            }
        }
        return VegaError::Success;
    }

    VegaError Graph::Build(std::string input_name, std::string output_name) {
        input_name_ = input_name;
        output_name_ = output_name;
        if (graph_state_ == GraphState::NeedInit) {
            VegaError error = Init();
            if (error != VegaError::Success) {
                LOG(ERROR) << "init graph failed";
                return error;
            }
        }
        if (graph_state_ == GraphState::Complete) {
            LOG(WARNING) << "graph is already built, skip build";
            return VegaError::Success;
        }

        // 构建计算图,每个节点指向所有下一个计算节点，保存在output_operators_map_
        for(const auto &current_operator : operators_list_) {
            for(const auto &output_name : current_operator->output_names_) {
                auto output_operator = operators_map_.find(output_name);
                if (output_operator != operators_map_.end()) {
                    current_operator->output_operators_map_.insert(std::make_pair(output_name, output_operator->second));
                }
            }
        }

        // 初始化节点的输入和输出空间
        VegaError error = OperatorUtils::InitOperatorInput(operators_list_);
        if (error != VegaError::Success) {
            LOG(ERROR) << "init operator input failed";
            return error;
        }
        error = OperatorUtils::InitOperatorOutput(pnnx_graph_->ops, operators_list_);
        if (error != VegaError::Success) {
            LOG(ERROR) << "init operator output failed";
            return error;
        }

        //  拓扑排序
        topo_operators_.clear();
        for (const auto &[_, op] : operators_map_) {
            // 根据输入节点构建拓扑排序
            if (op->type_ == "pnnx.Input" && !op->has_forward_) {
              this->ReverseTopo(op); // 根节点
            }
        }

        // 确保所有节点排序完成
        CHECK_EQ(topo_operators_.size(), operators_list_.size());
        // reverse topo_operators_
        std::reverse(topo_operators_.begin(), topo_operators_.end());

        graph_state_ = GraphState::Complete;

        // 析构pnnx_graph_
        if(pnnx_graph_ != nullptr) {
            pnnx_graph_.reset();
            pnnx_graph_ = nullptr;
        }
        return VegaError::Success;
    }

    void Graph::ReverseTopo(
        const std::shared_ptr<Operator> &root_op) {
      CHECK(root_op != nullptr) << "current operator is nullptr";
      root_op->has_forward_ = true;
      const auto &next_ops = root_op->output_operators_map_;
      for (const auto &[_, op] : next_ops) {
        if (op != nullptr) {
          if (!op->has_forward_) {
            this->ReverseTopo(op);
          }
        }
      }
      for (const auto &[_, op] : next_ops) {
        CHECK_EQ(op->has_forward_, true);
      }
      this->topo_operators_.push_back(root_op); // 递归出口，当节点出度为0的时候
    }
}//namespace vega_rt