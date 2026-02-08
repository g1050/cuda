#pragma once
#include "operator.hpp"
#include "operand.hpp"
#include "parameter.hpp"
#include "attribute.hpp"
#include "tensor.hpp"
#include "dtype.hpp"
#include "common.hpp"
#include <vector>
#include <memory>
#include <string>
#include <map>
#include <pnnx/ir.h>

namespace vega_rt {

    /**
     * @brief 计算图
     * 
     */
    class Graph {
        public:
            /**
             * @brief 计算图，对PNNXgraph的二次封装
             * 
             * @param param_path pnnx模型结构路径
             * @param bin_path pnnx权重路径
             */
            Graph(const std::string& param_path, const std::string& bin_path);
            ~Graph();
            
            /**
             * @brief 初始化，从PNNX计算图创建Vega计算图
             * 
             * @return VegaError 
             */
            VegaError Init();
            
            /**
             * @brief 获取Vega计算图的operators列表
             * 
             * @return const std::vector<OperatorSP>& 
             */
            const std::vector<OperatorSP>& operators() const;

             /**
              * @brief 构建计算图，只支持单输入单输出
              * 
              * @param input_name 
              * @param output_name 
              * @return VegaError 
              */
             VegaError Build(std::string input_name, std::string output_name);

             /**
             * @brief 获取拓扑排序后的队列
             * 
             * @return const std::vector<OperatorSP>& 
             */
            const std::vector<OperatorSP>& get_topo_queues() const{return topo_operators_;};
        private:
            std::string param_path_;
            std::string bin_path_;

            std::unique_ptr<pnnx::Graph> pnnx_graph_; // pnnx计算图

            std::map<std::string, OperatorSP> operators_map_; // 节点名称:VegaOperator
            std::vector<OperatorSP> operators_list_; // 节点列表

            std::string input_name_;
            std::string output_name_;

            enum class GraphState {
                NeedInit = -2, //pnnx->vegaRt
                NeedBuild = -1, //
                Complete = 0,
            };
            GraphState graph_state_ = GraphState::NeedInit; // 计算图状态

            std::vector<OperatorSP> topo_operators_; // 拓扑排序后的节点列表
        private:
            /**
             * @brief 初始化输入操作数
             * 
             * @param operator_sp Vega计算节点
             * @param pnnx_operands pnnx输入操作数
             * @return VegaError 
             */
            VegaError InitGraphInputOperands(OperatorSP operator_sp, std::vector<pnnx::Operand*> pnnx_operands);
            /**
             * @brief 初始化输出操作数
             * 
             * @param operator_sp Vega计算节点
             * @param pnnx_operands pnnx输出操作数
             * @return VegaError 
             */
            VegaError InitGraphOutputOperands(OperatorSP operator_sp, std::vector<pnnx::Operand*> pnnx_operands);
            /**
             * @brief 初始化参数,相较于PNNX用一个对象保存所有类型的参数(冗余数据)，该Paramter使用派生类表示特定的数据类型
             * 
             * @param operator_sp 
             * @param params 
             * @return VegaError 
             */
            VegaError InitGraphParams(OperatorSP operator_sp, std::map<std::string, pnnx::Parameter> params);
            /**
             * @brief 初始化权重信息
             * 
             * @param operator_sp 
             * @param attrs 
             * @return VegaError 
             */
            VegaError InitGraphAttrs(OperatorSP operator_sp, std::map<std::string, pnnx::Attribute> attrs);

            /**
             * @brief 拓扑排序
             * 
             * @param root_op 根节点
             */
            void ReverseTopo(const std::shared_ptr<Operator> &root_op);

            
        };
}