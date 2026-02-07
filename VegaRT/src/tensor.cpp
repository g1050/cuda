#include "tensor.hpp"
#include "common.hpp"
#include <algorithm>
#include <iostream>
namespace vega_rt {

    Tensor<float>::Tensor(uint32_t size) : raw_shape_(std::vector<uint32_t>{size}){
        data_ = arma::fcube(1, size, 1); // h,w,c
    }

    Tensor<float>::Tensor(uint32_t rows, uint32_t cols) : raw_shape_(std::vector<uint32_t>{rows,cols}){
        data_ = arma::fcube(rows, cols, 1);
    }

    Tensor<float>::Tensor(uint32_t channels, uint32_t rows, uint32_t cols) {//c,h,w
        // 压缩维度
        data_ = arma::fcube(rows, cols, channels);
        if(channels == 1 && rows == 1){
            raw_shape_ = std::vector<uint32_t>{cols};
        }else if(channels == 1){
            raw_shape_ = std::vector<uint32_t>{rows,cols};
        }else{
            raw_shape_ = std::vector<uint32_t>{channels,rows,cols};
        }
    }

    float Tensor<float>::at(uint32_t channel, uint32_t row, uint32_t col) const { 
        CHECK_LT(channel, raw_shape_[0]);
        CHECK_LT(row, raw_shape_[1]);
        CHECK_LT(col, raw_shape_[2]);
        return data_(row, col, channel); // cube: h,w,c , Tensor: c,h,w
    }

    void Tensor<float>::Fill(const std::vector<float>& values, bool row_major){
        CHECK_EQ(values.size(), data_.n_elem);
        int channels = raw_shape_[0], rows = raw_shape_[1], cols = raw_shape_[2];
        if(row_major){
            for(uint32_t i = 0; i < channels; i++){ // channel
                auto start_index = values.begin() + i * rows * cols;
                auto length = rows * cols;
                arma::fmat mat(cols, rows);
                std::copy(start_index, start_index + length, mat.begin());
                data_.slice(i) = mat.t(); // transpose，提前转置
            }
        }else{
            std::copy(values.begin(), values.end(), data_.begin());
        }
    }

    void Tensor<float>::show_shape() const {
        LOG(WARNING) << "shape: " << raw_shape_[0] << "x" << raw_shape_[1] << "x" << raw_shape_[2];
    }
}