cmake_minimum_required(VERSION 3.15)

project(matrix_sparse)
add_executable(matrix_sparse matrix_sparse.cpp) 

find_package(Ginkgo 1.7.0 EXACT REQUIRED)
target_link_libraries(matrix_sparse PUBLIC Ginkgo::ginkgo)
