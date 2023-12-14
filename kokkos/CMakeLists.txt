cmake_minimum_required(VERSION 3.16)

project(matrix_multiplication VERSION 0.1.0)

add_subdirectory(kokkos)
add_subdirectory(kokkos-kernels)

add_executable(hand_gemv hand_gemv.cpp)
target_link_libraries(hand_gemv Kokkos::kokkos)

add_executable(blas_gemv blas_gemv.cpp)
target_link_libraries(blas_gemv Kokkos::kokkos Kokkos::kokkoskernels)

add_executable(flux flux.cpp)
target_link_libraries(flux Kokkos::kokkos)
