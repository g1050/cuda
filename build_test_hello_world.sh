

RELEASE_MODE=$1
mkdir bin
# Debug模式
if [ "$RELEASE_MODE" == "debug" ]; then
    echo "start building hello_world!"
    nvcc -g -G -O0 -o bin/hello hello_world.cu -arch=sm_86
    # -g CPU调试
    # -G GPU调试
    # O0 优化级别，关闭优化
    # 指定目标 GPU 架构为 SM 86，Ampere,RTX 3060/3070/3080、A100 等
    echo "build hello_world finished!"
    ./bin/hello
elif [ "$RELEASE_MODE" == "release" ]; then
# Release模式
nvcc -O3 -o bin/hello hello_world.cu -arch=sm_86
./bin/hello
else
    echo "invalid release mode!"
    exit 1
fi