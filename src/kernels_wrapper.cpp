/**
 * @file kernels_wrapper.cpp
 * @brief C++ wrapper for 256-bit integer CUDA operations
 * 
 * This file provides a clean C++ interface to the CUDA kernels,
 * making the library easier to use from pure C++ code and other languages.
 */

#include "256bit/256bit.h"
#include <torch/torch.h>
#include <stdexcept>

namespace bit256 {

/**
 * @brief Compares two 256-bit integer tensors
 * 
 * @param a First 256-bit integer tensor (shape: [rows, 8])
 * @param b Second 256-bit integer tensor (shape: [rows, 8])
 * @param o Output tensor containing comparison results (shape: [rows])
 * 
 * @throws std::runtime_error if tensors have invalid dimensions or CUDA errors occur
 */
void compare(const torch::Tensor& a, const torch::Tensor& b, torch::Tensor& o) {
    // Validate input tensors
    if (a.dim() != 2 || b.dim() != 2 || o.dim() != 1) {
        throw std::runtime_error("Invalid tensor dimensions for comparison");
    }
    if (a.size(1) != 8 || b.size(1) != 8) {
        throw std::runtime_error("Tensors must have 8 columns for 256-bit integers");
    }
    if (a.size(0) != b.size(0) || a.size(0) != o.size(0)) {
        throw std::runtime_error("Tensor row counts must match");
    }
    
    compare_tensors_cuda(a, b, o);
}

/**
 * @brief Adds two 256-bit integer tensors with carry handling
 * 
 * @param a First operand tensor (shape: [rows, 8])
 * @param b Second operand tensor (shape: [rows, 8])
 * @param o Output tensor for result (shape: [rows, 8])
 * @param c Carry output tensor (shape: [rows])
 * 
 * @throws std::runtime_error if tensors have invalid dimensions or CUDA errors occur
 */
void add(const torch::Tensor& a, const torch::Tensor& b, torch::Tensor& o, torch::Tensor& c) {
    // Validate input tensors
    if (a.dim() != 2 || b.dim() != 2 || o.dim() != 2 || c.dim() != 1) {
        throw std::runtime_error("Invalid tensor dimensions for addition");
    }
    if (a.size(1) != 8 || b.size(1) != 8 || o.size(1) != 8) {
        throw std::runtime_error("Tensors must have 8 columns for 256-bit integers");
    }
    if (a.size(0) != b.size(0) || a.size(0) != o.size(0) || a.size(0) != c.size(0)) {
        throw std::runtime_error("Tensor row counts must match");
    }
    
    add_tensors_cuda(a, b, o, c);
}

/**
 * @brief Subtracts two 256-bit integer tensors with borrow handling
 * 
 * @param a First operand tensor (shape: [rows, 8])
 * @param b Second operand tensor (shape: [rows, 8])
 * @param o Output tensor for result (shape: [rows, 8])
 * 
 * @throws std::runtime_error if tensors have invalid dimensions or CUDA errors occur
 */
void subtract(const torch::Tensor& a, const torch::Tensor& b, torch::Tensor& o) {
    // Validate input tensors
    if (a.dim() != 2 || b.dim() != 2 || o.dim() != 2) {
        throw std::runtime_error("Invalid tensor dimensions for subtraction");
    }
    if (a.size(1) != 8 || b.size(1) != 8 || o.size(1) != 8) {
        throw std::runtime_error("Tensors must have 8 columns for 256-bit integers");
    }
    if (a.size(0) != b.size(0) || a.size(0) != o.size(0)) {
        throw std::runtime_error("Tensor row counts must match");
    }
    
    sub_tensors_cuda(a, b, o);
}

/**
 * @brief Performs modular addition of 256-bit integers
 * 
 * @param a First operand tensor (shape: [rows, 8])
 * @param b Second operand tensor (shape: [rows, 8])
 * @param m Modulus tensor (shape: [rows, 8])
 * @return Result tensor (shape: [rows, 8]) containing (a + b) mod m
 * 
 * @throws std::runtime_error if tensors have invalid dimensions or CUDA errors occur
 */
torch::Tensor modular_add(const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& m) {
    // Validate input tensors
    if (a.dim() != 2 || b.dim() != 2 || m.dim() != 2) {
        throw std::runtime_error("Invalid tensor dimensions for modular addition");
    }
    if (a.size(1) != 8 || b.size(1) != 8 || m.size(1) != 8) {
        throw std::runtime_error("Tensors must have 8 columns for 256-bit integers");
    }
    if (a.size(0) != b.size(0) || a.size(0) != m.size(0)) {
        throw std::runtime_error("Tensor row counts must match");
    }
    
    return modular_add_cuda(a, b, const_cast<torch::Tensor&>(m));
}

} // namespace bit256 