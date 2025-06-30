/**
 * @file 256bit.h
 * @brief Main header for 256-bit integer operations library
 * @author Your Name
 * @date 2024
 * 
 * This header provides a clean C++ interface for 256-bit integer arithmetic
 * operations using CUDA acceleration and PyTorch tensors.
 */

#ifndef BIT256_H
#define BIT256_H

#include <torch/torch.h>

namespace bit256 {

/**
 * @brief Compares two 256-bit integer tensors
 * 
 * @param a First 256-bit integer tensor (shape: [rows, 8])
 * @param b Second 256-bit integer tensor (shape: [rows, 8])
 * @param o Output tensor containing comparison results (shape: [rows])
 * 
 * @throws std::runtime_error if tensors have invalid dimensions or CUDA errors occur
 * 
 * @details
 * This function compares two 256-bit integers element by element,
 * starting from the most significant digit. Returns 1 if a >= b, 0 otherwise.
 * 
 * @note Both input tensors must have the same number of rows and 8 columns.
 * @note The output tensor must be pre-allocated with the correct size.
 * 
 * @example
 * ```cpp
 * auto a = torch::zeros({10, 8});
 * auto b = torch::zeros({10, 8});
 * auto result = torch::zeros({10});
 * bit256::compare(a, b, result);
 * ```
 */
void compare(const torch::Tensor& a, const torch::Tensor& b, torch::Tensor& o);

/**
 * @brief Adds two 256-bit integer tensors with carry handling
 * 
 * @param a First operand tensor (shape: [rows, 8])
 * @param b Second operand tensor (shape: [rows, 8])
 * @param o Output tensor for result (shape: [rows, 8])
 * @param c Carry output tensor (shape: [rows])
 * 
 * @throws std::runtime_error if tensors have invalid dimensions or CUDA errors occur
 * 
 * @details
 * Performs element-wise addition of 256-bit integers. Each integer is
 * represented as 8 int32_t values. Carry propagation is handled automatically.
 * 
 * @note Input tensors must have matching dimensions
 * @note Output tensors must be pre-allocated
 * @note Uses CUDA kernels for parallel execution
 * 
 * @see subtract
 * @see modular_add
 */
void add(const torch::Tensor& a, const torch::Tensor& b, torch::Tensor& o, torch::Tensor& c);

/**
 * @brief Subtracts two 256-bit integer tensors with borrow handling
 * 
 * @param a First operand tensor (shape: [rows, 8])
 * @param b Second operand tensor (shape: [rows, 8])
 * @param o Output tensor for result (shape: [rows, 8])
 * 
 * @throws std::runtime_error if tensors have invalid dimensions or CUDA errors occur
 * 
 * @details
 * Performs element-wise subtraction of 256-bit integers. Each integer is
 * represented as 8 int32_t values. Borrow propagation is handled automatically.
 * 
 * @note Input tensors must have matching dimensions
 * @note Output tensors must be pre-allocated
 * @note Uses CUDA kernels for parallel execution
 * 
 * @see add
 * @see modular_add
 */
void subtract(const torch::Tensor& a, const torch::Tensor& b, torch::Tensor& o);

/**
 * @brief Performs modular addition of 256-bit integers
 * 
 * @param a First operand tensor (shape: [rows, 8])
 * @param b Second operand tensor (shape: [rows, 8])
 * @param m Modulus tensor (shape: [rows, 8])
 * @return Result tensor (shape: [rows, 8]) containing (a + b) mod m
 * 
 * @throws std::runtime_error if tensors have invalid dimensions or CUDA errors occur
 * 
 * @details
 * This function implements modular addition using the following algorithm:
 * 
 * 1. Addition Step:
 *    - Add a and b using the standard addition kernel
 *    - Store result in temporary tensor 'sum'
 *    - Capture carry output for overflow detection
 * 
 * 2. Comparison Step:
 *    - Compare 'sum' with modulus 'm'
 *    - Generate boolean mask indicating where sum >= m
 * 
 * 3. Conditional Subtraction:
 *    - Subtract 'm' from 'sum' where sum >= m
 *    - Store result in temporary tensor 'result'
 * 
 * 4. Final Selection:
 *    - Use PyTorch's where() function to select final result
 *    - Return 'result' where sum >= m, 'sum' otherwise
 * 
 * @note This operation requires multiple kernel launches with synchronization
 * @note Memory usage is approximately 4x the input size due to intermediate tensors
 * 
 * @see add
 * @see compare
 * @see subtract
 */
torch::Tensor modular_add(const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& m);

} // namespace bit256

#endif // BIT256_H 