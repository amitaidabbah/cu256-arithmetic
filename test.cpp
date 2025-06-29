#include <torch/torch.h>
#include "kernels.h"
#include <cuda_runtime.h>


// Helper function to print 256-bit numbers
void print_uint256(const torch::Tensor& tensor) {
    auto data = tensor.data_ptr<int32_t>();
    std::cout << "0x";
    for (int i = 7; i >= 0; --i) {
        std::cout << std::hex << std::setw(8) << std::setfill('0') << data[i];
    }
    std::cout << std::dec << std::endl;
}

void test_comp_256bit() {
    std::cout << "Starting Test Compare!" << std::endl;

    // Allocate tensors on the GPU
    torch::Tensor a = torch::zeros({2, 8}, torch::kInt32).cuda();
    torch::Tensor b = torch::zeros({2, 8}, torch::kInt32).cuda();
    torch::Tensor o = torch::zeros({2}, torch::kInt32).cuda();

    // Initialize the first row: a = [1, 1, ..., 1], b = [1, 1, ..., 1]
    for (int i = 0; i < 8; ++i) {
        a[0][i] = 1;
        b[0][i] = 1;
    }

    // Initialize the second row: a = [2, 2, ..., 2], b = [1, 1, ..., 1]
    for (int i = 0; i < 8; ++i) {
        a[1][i] = 2;
        b[1][i] = 1;
    }

    // Calculate the expected result for comparison
    torch::Tensor expected = torch::zeros({2}, torch::kInt32);
    for (size_t row = 0; row < 2; row++) {
        bool is_greater = false;

        // Compare each element from the most significant to the least significant
        for (int i = 7; i >= 0; --i) {
            if (a[row][i].item<int32_t>() > b[row][i].item<int32_t>()) {
                expected[row] = 1; // a > b
                is_greater = true;
                break;
            }
            if (a[row][i].item<int32_t>() < b[row][i].item<int32_t>()) {
                expected[row] = 0; // a < b
                is_greater = true;
                break;
            }
        }
        // If all elements are equal, set to 1
        if (!is_greater) {
            expected[row] = 1;
        }
    }

    // Perform comparison using the CUDA function
    compare_tensors_cuda(a, b, o);

    cudaDeviceSynchronize();

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in compare_tensors_cuda: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // Print inputs and outputs
    std::cout << "Input A:" << std::endl;
    print_uint256(a.cpu()[0]);
    print_uint256(a.cpu()[1]);
    std::cout << "Input B:" << std::endl;
    print_uint256(b.cpu()[0]);
    print_uint256(b.cpu()[1]);

    // Print GPU output and expected output
    std::cout << "GPU Output: " << o.cpu() << std::endl;
    std::cout << "Expected Output: " << expected.cpu() << std::endl;

    // Validate the results
    assert(torch::equal(o.cpu(), expected));
    std::cout << "Test passed for 256-bit compare using positive values!\n" << std::endl;
}

void test_add_256bit() {
    std::cout << "Starting Test Add!" << std::endl;

    torch::Tensor a = torch::zeros({1, 8}, torch::kInt32).cuda();
    torch::Tensor b = torch::zeros({1, 8}, torch::kInt32).cuda();
    torch::Tensor o = torch::zeros({1, 8}, torch::kInt32).cuda();
    torch::Tensor c = torch::zeros({1}, torch::kInt32).cuda();

    for (int i = 0; i < 8; ++i) {
        a[0][i] = 0x7FFFFFFF;
    }

    // Set b = [1, 1, ..., 1]
    for (int i = 0; i < 8; ++i) {
        b[0][i] = 1;
    }

    // Calculate the expected result (a + b) with carry handling
    torch::Tensor expected = torch::zeros({1, 8}, torch::kInt32);
    int carry = 0;
    for (int i = 0; i < 8; ++i) {
        uint64_t sum = static_cast<uint64_t>(a[0][i].item<int32_t>()) + 
                       static_cast<uint64_t>(b[0][i].item<int32_t>()) + carry;
        carry = sum > UINT32_MAX ? 1 : 0;
        expected[0][i] = static_cast<int32_t>(sum);
    }

    add_tensors_cuda(a, b, o, c);

    std::cout << "Input A:" << std::endl;
    print_uint256(a.cpu()[0]);
    std::cout << "Input B:" << std::endl;
    print_uint256(b.cpu()[0]);

    std::cout << "GPU Output:" << std::endl;
    print_uint256(o.cpu()[0]);
    std::cout << "Expected Output:" << std::endl;
    print_uint256(expected[0]);

    int gpu_carry = c.cpu().item<int32_t>();
    std::cout << "GPU Output Carry C: " << gpu_carry << std::endl;
    std::cout << "Expected Carry: " << carry << std::endl;

    assert(torch::equal(o.cpu(), expected) && gpu_carry == carry);
    std::cout << "Test passed for 256-bit addition using positive values!\n" << std::endl;
}

void test_sub_256bit() {
    std::cout << "Starting Test Sub!" << std::endl;

    torch::Tensor a = torch::zeros({1, 8}, torch::kInt32).cuda();
    torch::Tensor b = torch::zeros({1, 8}, torch::kInt32).cuda();
    torch::Tensor o = torch::zeros({1, 8}, torch::kInt32).cuda();

    // Set a = [1, 2, 3, 4, 5, 6, 7, 8]
    for (int i = 0; i < 8; ++i) {
        a[0][i] = i + 1;
    }

    // Set b = [10, 20, 30, 40, 50, 60, 70, 80]
    for (int i = 0; i < 8; ++i) {
        b[0][i] = (i + 1) * 10;
    }

    // Expected result: (b - a)
    torch::Tensor expected = torch::zeros({1, 8}, torch::kInt32);
    for (int i = 0; i < 8; ++i) {
        expected[0][i] = (b[0][i].item<int32_t>() - a[0][i].item<int32_t>());
    }

    sub_tensors_cuda(b, a, o);

    std::cout << "Input A:" << std::endl;
    print_uint256(a.cpu()[0]);
    std::cout << "Input B:" << std::endl;
    print_uint256(b.cpu()[0]);
    std::cout << "GPU Output:" << std::endl;
    print_uint256(o.cpu()[0]);
    std::cout << "Expected Output:" << std::endl;
    print_uint256(expected[0]);

    assert(torch::equal(o.cpu(), expected));
    std::cout << "Test passed for 256-bit subtraction using positive values!\n" << std::endl;
}


void test_modular_add_256bit() {
    std::cout << "Starting Test Modular Add!" << std::endl;
    torch::Tensor a = torch::zeros({1, 8}, torch::kInt32).cuda();
    torch::Tensor b = torch::zeros({1, 8}, torch::kInt32).cuda();
    torch::Tensor m = torch::zeros({1, 8}, torch::kInt32).cuda();

    // Set a = [1, 2, 3, 4, 5, 6, 7, 8]
    for (int i = 0; i < 8; ++i) {
        a[0][i] = i + 1;
    }

    // Set b = [10, 20, 30, 40, 50, 60, 70, 80]
    for (int i = 0; i < 8; ++i) {
        b[0][i] = (i + 1) * 10;
    }

    // Set m = [100, 200, 300, 400, 500, 600, 700, 800]
    for (int i = 0; i < 8; ++i) {
        m[0][i] = (i + 1) * 100;
    }

    // Expected result: (a + b) % m
    torch::Tensor expected = torch::zeros({1, 8}, torch::kInt32);
    for (int i = 0; i < 8; ++i) {
        expected[0][i] = (a[0][i].item<int32_t>() + b[0][i].item<int32_t>()) % m[0][i].item<int32_t>();
    }

    torch::Tensor o_gpu = modular_add_cuda(a, b, m);

    std::cout << "Input A:" << std::endl;
    print_uint256(a.cpu()[0]);
    std::cout << "Input B:" << std::endl;
    print_uint256(b.cpu()[0]);
    std::cout << "Modulus M:" << std::endl;
    print_uint256(m.cpu()[0]);
    std::cout << "GPU Output:" << std::endl;
    print_uint256(o_gpu.cpu()[0]);
    std::cout << "Expected Output:" << std::endl;
    print_uint256(expected[0]);

    assert(torch::equal(o_gpu.cpu(), expected));
    std::cout << "Test passed for 256-bit integers modular add using positive values!\n" << std::endl;
}