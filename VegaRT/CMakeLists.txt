cmake_minimum_required(VERSION 3.18)
project(test_arma CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Armadillo REQUIRED)
find_package(glog REQUIRED)
find_package(GTest REQUIRED)

set(LIB_SOURCES
    src/tensor.cpp
)
set(EXEC_SOURCES
    test/test_tensor.cpp
)
set(link_lib glog::glog GTest::gtest)

add_library(tensor STATIC ${LIB_SOURCES})
target_include_directories(tensor PRIVATE include)
target_link_libraries(tensor PRIVATE ${ARMADILLO_LIBRARIES})

set(TEST_SOURCES
    test/test_tensor.cpp
)
add_executable(test_tensor ${TEST_SOURCES})
target_include_directories(test_tensor PRIVATE include)
target_link_libraries(test_tensor PRIVATE tensor)
target_link_libraries(test_tensor PRIVATE ${link_lib})

# 启用测试
enable_testing()
add_test(NAME TensorTests COMMAND test_tensor)


