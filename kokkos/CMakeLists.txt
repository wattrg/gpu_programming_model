cmake_minimum_required(VERSION 3.16)

project(matrix_multiplication VERSION 0.1.0)

add_subdirectory(kokkos)

add_executable(matrix_multiplication main.cpp)
target_link_libraries(matrix_multiplication Kokkos::kokkos)
