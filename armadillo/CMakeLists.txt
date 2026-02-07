cmake_minimum_required(VERSION 3.18)
project(test_arma CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(Armadillo REQUIRED)
find_package(glog REQUIRED)
find_package(GTest REQUIRED)

# 主程序
add_executable(matrix_demo test/main.cpp)
target_include_directories(matrix_demo PRIVATE ${ARMADILLO_INCLUDE_DIRS} 
                            include)
target_link_libraries(matrix_demo PRIVATE ${ARMADILLO_LIBRARIES})
target_link_libraries(matrix_demo PRIVATE glog::glog)



# 测试程序
add_executable(test_matrix_ops test/test_matrix_ops.cpp)
target_include_directories(test_matrix_ops PRIVATE ${ARMADILLO_INCLUDE_DIRS}
                            include)
target_link_libraries(test_matrix_ops PRIVATE ${ARMADILLO_LIBRARIES})
target_link_libraries(test_matrix_ops PRIVATE glog::glog)
target_link_libraries(test_matrix_ops PRIVATE GTest::gtest GTest::gtest_main)

# 启用测试
enable_testing()
add_test(NAME MatrixOpsTests COMMAND test_matrix_ops)

