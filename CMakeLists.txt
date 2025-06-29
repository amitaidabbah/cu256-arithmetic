cmake_minimum_required(VERSION 3.10)
project(pytorch_cpp_cuda_example)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optionally allow users to set LibTorch path
set(LIBTORCH_PATH "/usr/local/libtorch" CACHE PATH "Path to LibTorch")
set(CMAKE_PREFIX_PATH "${LIBTORCH_PATH}" CACHE PATH "Path to LibTorch" FORCE)

# Enable CUDA language
enable_language(CUDA)

# Find the Torch package
find_package(Torch REQUIRED)

# List of source files
set(SOURCES
    main.cpp
    test.cpp
    kernels.cu
)

# Add the executable
add_executable(example ${SOURCES})

# Enable CUDA separable compilation
set_target_properties(example PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link against Torch and CUDA libraries
target_link_libraries(example "${TORCH_LIBRARIES}" cudart)

# Include Torch and CUDA headers
target_include_directories(example PRIVATE 
    ${TORCH_INCLUDE_DIRS} 
    ${CUDA_INCLUDE_DIRS}
    "${LIBTORCH_PATH}/include"
    "${LIBTORCH_PATH}/include/torch/csrc/api/include"
)

# Set CUDA architecture (example for NVIDIA Turing architecture)
set_target_properties(example PROPERTIES CUDA_ARCHITECTURES "75")

# Set compile options for optimization
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(example PRIVATE -O3 -Wall -Wextra)
    target_compile_options(example PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler -Wall>)
endif()

# Set compile definitions for Torch (required for some systems)
target_compile_definitions(example PRIVATE TORCH_USE_CUDA)
