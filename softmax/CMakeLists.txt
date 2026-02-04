cmake_minimum_required(VERSION 3.18)

project(softmax LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# cmake -DSOFTMAX_ENABLE_CUDA=OFF
option(SOFTMAX_ENABLE_CUDA "Build CUDA implementation if available" ON)

# 定义静态库
add_library(softmax STATIC
  src/softmax_cpu.cpp
)


# 任何链接 softmax 的 target，都会自动包含头文件路径
target_include_directories(softmax
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

if(SOFTMAX_ENABLE_CUDA)
  add_library(vec_add STATIC
    src/vec_add.cu
  )
  # 头文件暴露
  target_include_directories(vec_add
    PUBLIC
      "${CMAKE_CURRENT_SOURCE_DIR}/include"
  )
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    # 添加cu源文件
    target_sources(softmax PRIVATE
      src/softmax.cu
    )

    set_target_properties(softmax vec_add PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
    )

    # Link cudart when CUDAToolkit is discoverable (optional).
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
      target_link_libraries(softmax PUBLIC CUDA::cudart)
      target_link_libraries(vec_add PUBLIC CUDA::cudart)
    endif()


  else()
    message(STATUS "CUDA compiler not found; building CPU-only softmax.")
  endif()
endif()

# 构建可执行文件（.cu 由 NVCC 编译，才能识别 <<<>>> kernel 启动语法）
add_executable(softmax_test test.cu)
target_link_libraries(softmax_test PRIVATE softmax vec_add)
# # 构建可执行文件
# add_executable(softmax_test test.cpp)
# target_link_libraries(softmax_test PRIVATE softmax vec_add)

