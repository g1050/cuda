# libpnnx - 仅 IR 与 store_zip 的独立库（不依赖 PyTorch）
cmake_minimum_required(VERSION 3.10)
project(pnnx CXX)

set(CMAKE_CXX_STANDARD 17)

add_library(pnnx SHARED
  ir.cpp
  store_zip.cpp
)

target_include_directories(pnnx
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

install(TARGETS pnnx
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES
    store_zip.hpp
    DESTINATION include/pnnx
)

# 不定义 BUILD_PNNX，避免依赖 PyTorch
# target_compile_definitions(pnnx PRIVATE BUILD_PNNX)
