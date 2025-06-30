cmake_minimum_required(VERSION 3.18)
project(256bit LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_EXTENSIONS OFF)

# -----------------------------------------------------------------------------
# Options
option(BUILD_SHARED_LIBS "Build shared instead of static" ON)
option(  "Build example executable" ON)

# Let users say: cmake -DLIBTORCH_PATH=/opt/libtorch ..
set(LIBTORCH_PATH "/usr/local/libtorch" CACHE PATH "Path to LibTorch")
list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_PATH}")

# Find dependencies
find_package(Torch REQUIRED)
enable_language(CUDA)

# -----------------------------------------------------------------------------
# Library target
add_library(256bit
    src/kernels.cu
    src/kernels_wrapper.cpp
)

# Set library properties
set_target_properties(256bit PROPERTIES
    VERSION 1.0.0
    SOVERSION 1
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "75"  # Adjust for your target GPU
)

# Include directories
target_include_directories(256bit
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${TORCH_INCLUDE_DIRS} 
        ${CUDA_INCLUDE_DIRS}
        "${LIBTORCH_PATH}/include"
        "${LIBTORCH_PATH}/include/torch/csrc/api/include"
)

# Link libraries
target_link_libraries(256bit
    PUBLIC 
        Torch::Torch 
        cudart
)

# Compile definitions
target_compile_definitions(256bit 
    PRIVATE 
        TORCH_USE_CUDA
)

# Compile options
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(256bit PRIVATE 
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wextra>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler -Wall>
    )
endif()

# -----------------------------------------------------------------------------
# Example executable (optional)
if(BUILD_EXAMPLES)
    add_executable(example 
        tests/main.cpp 
        tests/test.cpp
    )
    
    target_link_libraries(example PRIVATE 256bit)
    
    # Set compile options for example
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_options(example PRIVATE -O3 -Wall -Wextra)
    endif()

    # Simple example executable
    add_executable(simple_example examples/simple_example.cpp)
    target_link_libraries(simple_example PRIVATE 256bit)
    
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_options(simple_example PRIVATE -O3 -Wall -Wextra)
    endif()
endif()

# -----------------------------------------------------------------------------
# Install rules
include(GNUInstallDirs)

# Install library
install(TARGETS 256bit
        EXPORT 256bitTargets
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

# Install headers
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Install example if built
if(BUILD_EXAMPLES)
    install(TARGETS example simple_example
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

# Export targets
install(EXPORT 256bitTargets
        FILE 256bitTargets.cmake
        NAMESPACE 256bit::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/256bit)

# Create and install config files
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/256bitConfigVersion.cmake"
        VERSION 1.0.0
        COMPATIBILITY AnyNewerVersion)

configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/256bitConfig.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/256bitConfig.cmake"
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/256bit
        PATH_VARS CMAKE_INSTALL_INCLUDEDIR)

install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/256bitConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/256bitConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/256bit)
