#pragma once
#include "../layer.hpp"
#include <cstdint>
#include "../layer_factory.hpp"
#include "../../utils.hpp"
namespace vega_rt {
    class ConvLayer : public ParameterLayer {
        public:
            explicit ConvLayer(uint32_t output_channels, uint32_t input_channels, 
                uint32_t kernel_h, uint32_t kernel_w, 
                uint32_t padding_h, uint32_t padding_w,
                uint32_t stride_h, uint32_t stride_w,
                uint32_t groups,bool use_bias = true);
            
            VegaError Forward(const std::vector<TensorSP> &inputs, std::vector<TensorSP> &outputs) override;
            static VegaError GetInstance(
                const std::shared_ptr<Operator>& op,
                std::shared_ptr<Layer>& conv_layer);
        private:
            /**
             * @brief 检查输入、输出、参数、权重、偏置是否合法
             * 
             * @param inputs 输入
             * @param outputs 输出
             * @return VegaError 
             */
            VegaError Check(const std::vector<TensorSP> &inputs, const std::vector<TensorSP> &outputs);

            /**
             * @brief Image to column，将卷积运算转换为矩阵乘法
             * 
             * @param input 输入张量
             * @param kernel_w 卷积核宽度
             * @param kernel_h 卷积核高度
             * @param input_w 输入宽度
             * @param input_h 输入高度
             * @param input_c_group 输入通道组数
             * @param group 分组
             * @param row_len 行长度
             * @param col_len 列长度
             * @return arma::fmat 输出矩阵
             */
            arma::fmat Im2Col(TensorSP input, uint32_t kernel_w,
                uint32_t kernel_h, uint32_t input_w,
                uint32_t input_h, uint32_t input_c_group,
                uint32_t group, uint32_t row_len,
                uint32_t col_len) const;
            
            /**
             * @brief 初始化卷积核到frowvec
             * 
             */
            void InitIm2ColWeight();

            std::vector<arma::frowvec> kernel_matrix_arr_;  // frowvec:float row vector，单精度行向量
            uint32_t output_channels_;
            uint32_t input_channels_;
            uint32_t kernel_h_;
            uint32_t kernel_w_;
            uint32_t padding_h_;
            uint32_t padding_w_;
            uint32_t stride_h_;
            uint32_t stride_w_;
            uint32_t groups_;
            bool use_bias_;
    };
    using ConvLayerSP = std::shared_ptr<ConvLayer>;
}