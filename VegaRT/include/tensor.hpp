#pragma once
#include <armadillo>
#include <cstdint>
#include <memory>
#include <vector>

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

            std::vector<uint32_t> raw_shape() const { return raw_shape_; }
            float at(uint32_t channel, uint32_t row, uint32_t col) const ;
            void Fill(const std::vector<float>& values, bool row_major) ;
            uint32_t size() const { return data_.n_elem; }
            void show_shape() const ;
        private:    
            std::vector<uint32_t> raw_shape_;
            arma::fcube data_;
    };

}