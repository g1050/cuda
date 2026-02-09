#include "layer/layer_factory.hpp"
#include <glog/logging.h>
namespace vega_rt {
    void LayerRegisterer::RegisterCreator(const std::string &layer_type,
        const Creator &creator) {
        CHECK(creator != nullptr);
        // 获取全局算子注册表
        CreateRegistry& registry = Registry();
        CHECK_EQ(registry.count(layer_type), 0)
            << "Layer type: " << layer_type << " has already registered!";
        // 插入注册表
        registry.insert({layer_type, creator});
    }

    LayerRegisterer::CreateRegistry& LayerRegisterer::Registry() {
        static CreateRegistry* kRegistry = new CreateRegistry();
        CHECK(kRegistry != nullptr) << "Global layer register init failed!";
        return *kRegistry;
    }

    LayerSP LayerRegisterer::CreateLayer(const OperatorSP &op) {
        CreateRegistry& registry = Registry();
        const auto op_type = op->type_;
        CHECK(registry.count(op_type) != 0)
            << "Layer type: " << op_type << " not registered!";
        // 找到creator函数
        const auto creator = registry.at(op_type);
        LayerSP layer ;
        if (creator(op, layer) != VegaError::Success) {
            LOG(ERROR) << "Create layer failed for type: " << op_type;
            return nullptr;
        }
        return layer;
    }


}