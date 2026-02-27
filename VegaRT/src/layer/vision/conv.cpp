#include "layer/vision/conv.hpp"
#include <cstdint>
namespace vega_rt {
    ConvLayer::ConvLayer(uint32_t output_channels, uint32_t input_channels, 
        uint32_t kernel_h, uint32_t kernel_w, 
        uint32_t padding_h, uint32_t padding_w,
        uint32_t stride_h, uint32_t stride_w,
        uint32_t groups,bool use_bias) : ParameterLayer("nn.Conv2d"),
        output_channels_(output_channels), input_channels_(input_channels),
        kernel_h_(kernel_h), kernel_w_(kernel_w),
        padding_h_(padding_h), padding_w_(padding_w),
        stride_h_(stride_h), stride_w_(stride_w),
        groups_(groups), use_bias_(use_bias) {
            // 分组卷积
            if(groups_ != 1){ // 修改了输入通道数
                input_channels_ /= groups_;
            }
            this->InitWeightParameters(output_channels_, input_channels_, kernel_h_, kernel_w_);
            if(use_bias_){
                this->InitBiasParameters(output_channels_, 1, 1, 1);
            }
    }

    VegaError ConvLayer::Check(const std::vector<TensorSP> &inputs, const std::vector<TensorSP> &outputs) {
        if(inputs.empty()){
            LOG(ERROR) << "inputs is empty";
            return VegaError::InvalidParam;
        }
        if(inputs.size() != outputs.size()){
            LOG(ERROR) << "inputs and outputs size mismatch";
            return VegaError::InvalidParam;
        }
        if(weight_tensors_.empty()){
            LOG(ERROR) << "weight_tensors_ is empty";
            return VegaError::InvalidParam;
        }

        if(use_bias_ && bias_tensors_.size() != weight_tensors_.size()){
            LOG(ERROR) << "bias_tensors_ size mismatch";
            return VegaError::InvalidParam;
        }

        if(stride_h_ <= 0 || stride_w_ <= 0){
            LOG(ERROR) << "stride_h and stride_w must be greater than 0";
            return VegaError::InvalidParam;
        }

        if(padding_h_ < 0 || padding_w_ < 0){
            LOG(ERROR) << "padding_h and padding_w must be greater than 0";
            return VegaError::InvalidParam;
        }
        return VegaError::Success;
    }

    arma::fmat ConvLayer::Im2Col(TensorSP input, uint32_t kernel_w,
        uint32_t kernel_h, uint32_t input_w,
        uint32_t input_h, uint32_t input_c_group,
        uint32_t group, uint32_t row_len,
        uint32_t col_len) const {
        // 例如对于input (4,4)，padding (0,0),kernel size (3,3)，stride (1,1)，
        // 则输出的矩阵大小应该是 (4+2*0-3)/1+1)*(4+2*0-3)/1+1) = 2*2, 即2*2的矩阵
        // 这里的input_matrix是将滑窗需要的矩阵复制，避免滑窗，
        // 每行的长度等于卷积核的大小row_len=kernel_h * kernel_w=3*3=9,
        // 行数等于，滑窗移动的次数，即输出元素的个数，col_len=output_h * output_w=2*2=4
        arma::fmat input_matrix(row_len*input_c_group, col_len); //多通道展平
        
        // padding后的输入矩阵大小
        const uint32_t input_padded_h = input_h + 2 * padding_h_;
        const uint32_t input_padded_w = input_w + 2 * padding_w_;
        const float padding_value = 0.0f; // 填充值

        for(uint32_t ic = 0;ic<input_c_group;ic++){//分组，每组的输入通道数
            float* input_channel_ptr =
                input->matrix_raw_ptr(ic + group * input_c_group);
            uint32_t current_col = 0;
            uint32_t channel_row = ic * row_len;
            // 按照列滑动
            for (uint32_t w = 0; w <= input_padded_w - kernel_w; w += stride_w_) {
                for (uint32_t r = 0; r <= input_padded_h - kernel_h; r += stride_h_) {
                    float* input_matrix_ptr =
                        input_matrix.colptr(current_col) + channel_row; // 列主序，所以一列一个patch,按照通道拼接
                    // 按照列展开
                    for (uint32_t kw = 0; kw < kernel_w; ++kw) {
                        for (uint32_t kh = 0; kh < kernel_h; ++kh) {
                            const uint32_t logical_row = r + kh; // 输入矩阵上的位置+kernel内的位置
                            const uint32_t logical_col = w + kw;
                            const bool in_bounds =
                                (logical_row >= padding_h_ &&
                                 logical_row < padding_h_ + input_h &&
                                 logical_col >= padding_w_ &&
                                 logical_col < padding_w_ + input_w);
                            // 如果有值，则赋值，没有值则padding
                            if (in_bounds) {
                                const uint32_t in_row = logical_row - padding_h_;
                                const uint32_t in_col = logical_col - padding_w_;
                                // Armadillo 列主序: (row,col) -> row + col * input_h
                                *input_matrix_ptr =
                                    *(input_channel_ptr + in_row + in_col * input_h);
                            } else {
                                *input_matrix_ptr = padding_value;
                            }
                            ++input_matrix_ptr;
                        }
                    }
                    ++current_col;
                }
            }
        }

        // 常见框架习惯：一行一个 patch（一个输出位置的 receptive field），转置便于后续矩阵乘法
        // return input_matrix.t();
        return input_matrix;
    }

    void ConvLayer::InitIm2ColWeight() {
        const uint32_t kernel_count = weight_tensors_.size();
        CHECK(kernel_count > 0) << "kernel count must greater than zero";
        const uint32_t kernel_h = weight_tensors_.at(0)->rows();
        const uint32_t kernel_w = weight_tensors_.at(0)->cols();
        const uint32_t kernel_c = weight_tensors_.at(0)->channels();
        const uint32_t kernel_size = kernel_c * kernel_h * kernel_w;  // 每个卷积核展开后的大小
        CHECK(kernel_h > 0 && kernel_w > 0 && kernel_c > 0)
            << "The size of kernel matrix should be greater than zero";

        const uint32_t kernel_count_group = kernel_count / groups_;
        CHECK(kernel_count_group > 0) << "kernel_count_group must be greater than zero";

        for(uint32_t i=0;i<kernel_count;++i){
            CHECK(weight_tensors_.at(i)->channels() == kernel_c) << "kernel_c mismatch";
            CHECK(weight_tensors_.at(i)->channels() == input_channels_) << "kernel_c mismatch";
        }

        kernel_matrix_arr_.resize(groups_);
        for(uint32_t g = 0; g < groups_; ++g) {
            arma::frowvec& current_vect = kernel_matrix_arr_.at(g);
            current_vect.resize(kernel_count_group * kernel_size);

            // 填充权重数据：按组组织，每组 kernel_count_group 个卷积核
            uint32_t weight_idx = g * kernel_count_group;
            uint32_t offset = 0;
            for(uint32_t k = 0; k < kernel_count_group; ++k) {
                const TensorSP& kernel_tensor = weight_tensors_.at(weight_idx + k);
                // 将 3D 权重张量 (c,h,w) 展开为 1D 向量
                for(uint32_t c = 0; c < kernel_c; ++c) {
                    for(uint32_t h = 0; h < kernel_h; ++h) {
                        for(uint32_t w = 0; w < kernel_w; ++w) {
                            current_vect[offset++] = kernel_tensor->at(c, h, w);
                        }
                    }
                }
            }
        }

    }
    VegaError ConvLayer::Forward(const std::vector<TensorSP> &inputs, std::vector<TensorSP> &outputs) {
        CHECK(Check(inputs, outputs) == VegaError::Success) << "ConvLayer::Forward input, output, parameter, weight, bias check failed";

        const uint32_t batch_size = inputs.size();
        const uint32_t kernel_count = weight_tensors_.size();// 卷积核的数量，卷积核可能是多通道
        const uint32_t kernel_h = weight_tensors_.at(0)->rows();
        const uint32_t kernel_w = weight_tensors_.at(0)->cols();
        const uint32_t kernel_c = weight_tensors_.at(0)->channels();
        const uint32_t row_length = kernel_h * kernel_w;
        const uint32_t kernel_count_group = kernel_count / groups_; // 分组卷积
        CHECK(kernel_h > 0 && kernel_w > 0 && kernel_c > 0) << "kernel_h, kernel_w, kernel_c must be greater than 0";

        // check each kernel
        for(uint32_t i = 0; i < kernel_count; ++i) {
            CHECK(weight_tensors_.at(i)->rows() == kernel_h) << "kernel_h mismatch";
            CHECK(weight_tensors_.at(i)->cols() == kernel_w) << "kernel_w mismatch";
            CHECK(weight_tensors_.at(i)->channels() == kernel_c) << "kernel_c mismatch";
        }

        // 展开卷积核到frowvec
        if(kernel_matrix_arr_.empty()){
            for(uint32_t i = 0; i < kernel_count; ++i) {
                InitIm2ColWeight();
            }
        }

        for(uint32_t i = 0; i < batch_size; ++i) {
            const TensorSP &input = inputs.at(i);

            CHECK(input != nullptr && !input->empty()) << "input is nullptr or empty";
            const uint32_t input_c = input->channels();
            const uint32_t input_padded_h = input->rows() + 2 * padding_h_;
            const uint32_t input_padded_w = input->cols() + 2 * padding_w_;
            const uint32_t output_h = (input_padded_h - kernel_h) / stride_h_ + 1;
            const uint32_t output_w = (input_padded_w - kernel_w) / stride_w_ + 1;
            // 分组卷积，输出和输入通道需要满足可以平均分组的要求
            if(groups_ != 1){
                CHECK(kernel_count % groups_ == 0) << "kernel_count must be divisible by groups";
                CHECK(input_c % groups_ == 0) << "input_c must be divisible by groups";
            }
            const uint32_t input_c_group = input_c / groups_; // 分组，每组输入通道组数
            CHECK(input_c_group == kernel_c) << "The number of channel for the kernel "
                                                "matrix and input tensor do not match";

            const uint32_t col_len = output_h * output_w;
            CHECK(col_len > 0) << "col_len must be greater than 0";

            TensorSP &output = outputs.at(i);//按batch_size选择
            if (output == nullptr || output->empty()){
                output = TensorCreate(output_channels_, output_h, output_w);
            }

            // 按组处理：Im2Col 的 group 参数为当前组索引 (0 .. groups_-1)，不是分组总数
            for (uint32_t g = 0; g < groups_; ++g) {
                // 讲输入矩阵im2col展开
                const auto& input_matrix =
                    Im2Col(input, kernel_w, kernel_h, input->cols(), input->rows(),
                           input_c_group, g, row_length, col_len);
                LOG(ERROR) << "input_matrix: " << input_matrix; 
                 
                const uint32_t kernel_count_group_start = kernel_count_group * g;
                
            }
        }
        return VegaError::Success;
    }

    VegaError ConvLayer::GetInstance(const std::shared_ptr<Operator>& op, std::shared_ptr<Layer>& conv_layer) {
        CHECK(op != nullptr) << "Conv operator is nullptr";
        conv_layer = std::make_shared<ConvLayer>(1,1,1,1,1,1,1,1,1,true);
        return VegaError::Success;
    }
    LayerRegistererWrapper kConvGetInstance("nn.Conv2d", ConvLayer::GetInstance);

}