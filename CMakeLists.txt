cmake_minimum_required(VERSION 3.12)
project(page-rank-mpi CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 REQUIRED)

add_executable(page-rank src/main.cpp)
target_include_directories(page-rank PRIVATE 
    ${CMAKE_SOURCE_DIR}/external/fast_matrix_market/include)
target_link_libraries(page-rank 
  PRIVATE 
    Eigen3::Eigen
  )