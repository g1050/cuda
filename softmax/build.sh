# -S 源码目录 -B 构建目录
cmake -S ./ -B ./build -DSOFTMAX_ENABLE_CUDA=ON
# -j 并行构建
cmake --build ./build -j

build/softmax_test