#pragma once
#include <armadillo>
#include <cstdint>
#include <memory>
#include <glog/logging.h>
#include <vector>
#include <memory>
namespace vega_rt {

    template<typename T = float>
    class Tensor {};

    /**
     * @brief 张量类型
     * 
     * @tparam  
     */
    template <>
    class Tensor<float> {
        public:
            explicit Tensor() = default;
            /**
             * @brief Construct a new Tensor object
             * 
             * @param size 
             */
            explicit Tensor(uint32_t size);//必须显示构造
            explicit Tensor(uint32_t rows, uint32_t cols);
            explicit Tensor(uint32_t channels, uint32_t rows, uint32_t cols);//c,h,w存储
            explicit Tensor(const std::vector<uint32_t> &shapes);

            std::vector<uint32_t> raw_shape() const { return raw_shape_; }
            std::vector<uint32_t> shape() const { return {raw_shape_[0], raw_shape_[1], raw_shape_[2]}; }
            float at(uint32_t channel, uint32_t row, uint32_t col) const ;
            void Fill(const std::vector<float>& values, bool row_major) ;
            uint32_t size() const { return data_.n_elem; }
            void show_shape() const ;
            float index(uint32_t index) const { return data_.at(index); }

            /**
            * 返回张量中的数据
            * @return 张量中的数据
            */
            arma::fcube &data();
            
            /**
             * @brief 随机初始化张量
             * 
             */
            void Rand();

            /**
             * @brief 显示张量
             * 
             */
            void Show() const;
            bool empty() const { return this->data_.empty(); }
            uint32_t channels() const { return this->data_.n_slices; }
            uint32_t rows() const { return this->data_.n_rows; }
            uint32_t cols() const { return this->data_.n_cols; }
            std::vector<uint32_t> shapes() const ;
            arma::fmat &slice(uint32_t channel);

        private:    
            std::vector<uint32_t> raw_shape_;
            arma::fcube data_;
    };

    using TensorSP = std::shared_ptr<Tensor<float>>;


}