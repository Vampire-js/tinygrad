cmake_minimum_required(VERSION 3.16)

project(VampireGrad LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(VampireGrad
    src/main.cpp
    src/tensor.cpp
    src/matmul.cpp
    src/matmul_gpu.cu
)

target_include_directories(VampireGrad PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)