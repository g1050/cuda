cmake_minimum_required(VERSION 3.18)
project(vega_rt CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Armadillo REQUIRED)
find_package(glog REQUIRED)
find_package(GTest REQUIRED)

set(link_lib glog::glog GTest::gtest)

add_subdirectory(3rd_party)
file(GLOB LIB_SOURCES src/*.cpp)
file(GLOB TEST_SOURCES test/*.cpp)
add_library(vega_rt STATIC ${LIB_SOURCES})
target_include_directories(vega_rt PRIVATE include)
target_link_libraries(vega_rt PRIVATE ${link_lib} pnnx)
target_include_directories(vega_rt PUBLIC 3rd_party)

enable_testing()
foreach(test_src ${TEST_SOURCES})
    # 获取文件名，不含扩展名
    get_filename_component(test_name ${test_src} NAME_WE)
    # 生成可执行文件
    add_executable(${test_name} ${test_src})
    target_include_directories(${test_name} PRIVATE include)
    target_include_directories(${test_name} PRIVATE 3rd_party)
    target_link_libraries(${test_name} PRIVATE vega_rt)
    target_link_libraries(${test_name} PRIVATE ${link_lib})
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()



