# 256-bit Integer Operations Library

A high-performance CUDA library for 256-bit integer arithmetic operations using PyTorch tensors.

## Overview

This library provides GPU-accelerated 256-bit integer arithmetic operations including addition, subtraction, comparison, and modular arithmetic. Each 256-bit integer is represented as 8 int32_t values in a PyTorch tensor, enabling efficient batch processing of multiple integers.

## Features

- **High Performance**: CUDA kernels for parallel execution on GPU
- **PyTorch Integration**: Seamless integration with PyTorch tensors
- **Batch Processing**: Process multiple 256-bit integers simultaneously
- **Error Handling**: Comprehensive validation and error reporting
- **Easy Integration**: Clean C++ API with namespace organization

## Supported Operations

1. **Addition**: `bit256::add()` - Adds two 256-bit integers with carry handling
2. **Subtraction**: `bit256::subtract()` - Subtracts two 256-bit integers with borrow handling
3. **Comparison**: `bit256::compare()` - Compares two 256-bit integers
4. **Modular Addition**: `bit256::modular_add()` - Adds two 256-bit integers modulo a third integer

## Requirements

- CUDA Toolkit 11.0 or higher
- PyTorch C++ API (LibTorch)
- CMake 3.18 or higher
- C++17 compiler
- CUDA-capable GPU (Compute Capability 7.5 or higher)

## Installation

### Building from Source

```bash
# Clone the repository
git clone <repository-url>
cd 256bit

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake -DCMAKE_BUILD_TYPE=Release -DLIBTORCH_PATH=/path/to/libtorch ..

# Build the library and examples
cmake --build . -j$(nproc)

# Install (optional)
cmake --install . --prefix /usr/local
```

### CMake Options

- `BUILD_SHARED_LIBS`: Build shared instead of static library (default: ON)
- `BUILD_EXAMPLES`: Build example executables (default: ON)
- `LIBTORCH_PATH`: Path to LibTorch installation

## Usage

### Basic Usage

```cpp
#include "256bit/256bit.h"
#include <torch/torch.h>

int main() {
    // Create tensors on GPU (each 256-bit integer = 8 int32_t values)
    auto a = torch::zeros({1, 8}, torch::kInt32).cuda();
    auto b = torch::zeros({1, 8}, torch::kInt32).cuda();
    auto result = torch::zeros({1, 8}, torch::kInt32).cuda();
    auto carry = torch::zeros({1}, torch::kInt32).cuda();

    // Initialize values
    for (int i = 0; i < 8; ++i) {
        a[0][i] = 0x7FFFFFFF;  // Maximum 32-bit value
        b[0][i] = 1;
    }

    // Perform addition
    bit256::add(a, b, result, carry);
    
    // Check carry
    if (carry.cpu().item<int32_t>() > 0) {
        std::cout << "Overflow occurred!" << std::endl;
    }
}
```

### Batch Processing

```cpp
// Process multiple 256-bit integers at once
auto batch_size = 1000;
auto a = torch::zeros({batch_size, 8}, torch::kInt32).cuda();
auto b = torch::zeros({batch_size, 8}, torch::kInt32).cuda();
auto result = torch::zeros({batch_size, 8}, torch::kInt32).cuda();
auto carry = torch::zeros({batch_size}, torch::kInt32).cuda();

// Initialize batch data
// ... initialize a and b ...

// Process entire batch
bit256::add(a, b, result, carry);
```

### Modular Arithmetic

```cpp
auto a = torch::zeros({1, 8}, torch::kInt32).cuda();
auto b = torch::zeros({1, 8}, torch::kInt32).cuda();
auto modulus = torch::zeros({1, 8}, torch::kInt32).cuda();

// Initialize values
// ... initialize a, b, and modulus ...

// Compute (a + b) mod modulus
auto result = bit256::modular_add(a, b, modulus);
```

## API Reference

### `bit256::add(const torch::Tensor& a, const torch::Tensor& b, torch::Tensor& o, torch::Tensor& c)`

Adds two 256-bit integer tensors with carry handling.

**Parameters:**
- `a`: First operand tensor (shape: [rows, 8])
- `b`: Second operand tensor (shape: [rows, 8])
- `o`: Output tensor for result (shape: [rows, 8])
- `c`: Carry output tensor (shape: [rows])

**Throws:** `std::runtime_error` if tensors have invalid dimensions

### `bit256::subtract(const torch::Tensor& a, const torch::Tensor& b, torch::Tensor& o)`

Subtracts two 256-bit integer tensors with borrow handling.

**Parameters:**
- `a`: First operand tensor (shape: [rows, 8])
- `b`: Second operand tensor (shape: [rows, 8])
- `o`: Output tensor for result (shape: [rows, 8])

**Throws:** `std::runtime_error` if tensors have invalid dimensions

### `bit256::compare(const torch::Tensor& a, const torch::Tensor& b, torch::Tensor& o)`

Compares two 256-bit integer tensors.

**Parameters:**
- `a`: First operand tensor (shape: [rows, 8])
- `b`: Second operand tensor (shape: [rows, 8])
- `o`: Output tensor containing comparison results (shape: [rows])

**Returns:** 1 if a >= b, 0 otherwise

**Throws:** `std::runtime_error` if tensors have invalid dimensions

### `bit256::modular_add(const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& m)`

Performs modular addition of 256-bit integers.

**Parameters:**
- `a`: First operand tensor (shape: [rows, 8])
- `b`: Second operand tensor (shape: [rows, 8])
- `m`: Modulus tensor (shape: [rows, 8])

**Returns:** Result tensor (shape: [rows, 8]) containing (a + b) mod m

**Throws:** `std::runtime_error` if tensors have invalid dimensions

## Linking from Other Projects

### CMake Integration

```cmake
# Find the library
find_package(256bit CONFIG REQUIRED)

# Link to your target
add_executable(myapp main.cpp)
target_link_libraries(myapp PRIVATE 256bit::256bit)
```

### Manual Linking

If you installed the library manually:

```cmake
find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)

add_executable(myapp main.cpp)
target_link_libraries(myapp 
    PRIVATE 
        /path/to/lib256bit.a
        Torch::Torch 
        cudart
)
target_include_directories(myapp PRIVATE /path/to/include)
```

## Examples

The project includes several examples:

- `example`: Comprehensive test suite demonstrating all operations
- `simple_example`: Basic usage example showing addition and comparison

To build examples:
```bash
cmake -DBUILD_EXAMPLES=ON ..
make
```

## Performance

TBD

## Error Handling

The library provides comprehensive error checking:

- **Tensor Validation**: Checks for correct dimensions and data types
- **CUDA Error Detection**: Catches and reports CUDA kernel launch failures
- **Exception Safety**: All operations throw `std::runtime_error` on failure

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

TBD, currently free for all use.



