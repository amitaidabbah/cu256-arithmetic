/**
 * @file simple_example.cpp
 * @brief Simple example demonstrating 256-bit integer operations
 * 
 * This example shows how to use the 256bit library for basic arithmetic
 * operations on 256-bit integers represented as PyTorch tensors.
 */

#include "256bit/256bit.h"
#include <torch/torch.h>
#include <iostream>
#include <iomanip>

// Helper function to print 256-bit numbers
void print_uint256(const torch::Tensor& tensor) {
    auto data = tensor.data_ptr<int32_t>();
    std::cout << "0x";
    for (int i = 7; i >= 0; --i) {
        std::cout << std::hex << std::setw(8) << std::setfill('0') << data[i];
    }
    std::cout << std::dec << std::endl;
}

int main() {
    std::cout << "256-bit Integer Operations Example" << std::endl;
    std::cout << "==================================" << std::endl;

    // Create tensors on GPU
    auto a = torch::zeros({1, 8}, torch::kInt32).cuda();
    auto b = torch::zeros({1, 8}, torch::kInt32).cuda();
    auto result = torch::zeros({1, 8}, torch::kInt32).cuda();
    auto carry = torch::zeros({1}, torch::kInt32).cuda();

    // Initialize test values
    for (int i = 0; i < 8; ++i) {
        a[0][i] = 0x7FFFFFFF;  // Maximum 32-bit value
        b[0][i] = 1;
    }

    std::cout << "Input A:" << std::endl;
    print_uint256(a.cpu()[0]);
    std::cout << "Input B:" << std::endl;
    print_uint256(b.cpu()[0]);

    // Perform addition using the library
    try {
        cu256bit::add(a, b, result, carry);
        
        std::cout << "Result (A + B):" << std::endl;
        print_uint256(result.cpu()[0]);
        std::cout << "Carry: " << carry.cpu().item<int32_t>() << std::endl;
        
        std::cout << "Addition completed successfully!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error during addition: " << e.what() << std::endl;
        return 1;
    }

    // Test comparison
    auto compare_result = torch::zeros({1}, torch::kInt32).cuda();
    try {
        cu256bit::compare(a, b, compare_result);
        std::cout << "Comparison result (A >= B): " << compare_result.cpu().item<int32_t>() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error during comparison: " << e.what() << std::endl;
        return 1;
    }

    return 0;
} 