#pragma once
#include <map>
#include <string>
#include "../operator.hpp"
#include "layer.hpp"
#include "../common.hpp"
namespace vega_rt {
    class LayerRegisterer {
        public:
            /**
            * @brief 创建Layer的函数指针,参数为算子类和算子实例
            * 
            */
            typedef VegaError (*Creator)(
                const OperatorSP &op,
                LayerSP &layer);
            /**
            * @brief 算子注册器,map类型
            * 
            */
            typedef std::map<std::string, Creator> CreateRegistry;
        public:
            /**
            * 向注册表注册算子
            * @param layer_type 算子的类型
            * @param creator 需要注册算子的注册表
            */
            static void RegisterCreator(const std::string &layer_type,
                                        const Creator &creator);
        
            /**
            * 通过算子参数op来初始化Layer,算子实例
            * @param op 保存了初始化Layer信息的算子
            * @return 初始化后的Layer
            */
            static LayerSP CreateLayer(
                const OperatorSP &op);
        
            /**
            * 获取全局算子注册表
            * @return 算子注册表
            */
            static CreateRegistry &Registry();
        
            // /**
            // * 返回所有已被注册算子的类型
            // * @return 注册算子的类型列表
            // */
            // static std::vector<std::string> layer_types();
       };

       class LayerRegistererWrapper {
        public:
         LayerRegistererWrapper(const std::string &layer_type,
                                const LayerRegisterer::Creator &creator) {
           LayerRegisterer::RegisterCreator(layer_type, creator);
         }
    };
}

