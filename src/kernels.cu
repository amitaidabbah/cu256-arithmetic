#include <torch/torch.h>
#include <torch/types.h>
#include <iostream>

// Forward declarations for CUDA functions
void compare_tensors_cuda(
    const torch::Tensor& a, 
    const torch::Tensor& b, 
    torch::Tensor& o);

torch::Tensor modular_add_cuda(
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& m
);

void add_tensors_cuda(
    const torch::Tensor& a, 
    const torch::Tensor& b, 
    torch::Tensor& o,
    torch::Tensor& c
);

void sub_tensors_cuda(
    const torch::Tensor& a, 
    const torch::Tensor& b, 
    torch::Tensor& o
);

__global__ void compare_kernel_32_256(
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> a,
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> b,
    torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> o
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int rows = a.size(0);
    int cols = a.size(1);
    if (row < rows) {
        for (int i = 7; i >= 0; --i) {
            if (a[row][i] > b[row][i]) {
                o[row] = 1;
                return;
            }
            if (a[row][i] < b[row][i]) {
                o[row] = 0;
                return;
            }
        }
        o[row] = 1;
    }
}

__global__ void add_kernel_32_256(
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> a,
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> b,
    torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> o,
    torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> c
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int rows = a.size(0);
    int cols = a.size(1);

    // Check if the row is within the bounds of the tensor
    if (row < rows) 
    {
        int carry = 0;
        for (int i = 0; i < cols; ++i) {
            uint64_t sum = static_cast<uint64_t>(a[row][i]) + 
                                     static_cast<uint64_t>(b[row][i]) +
                                     static_cast<uint64_t>(carry);
            carry = sum > UINT_MAX ? 1 : 0;
            o[row][i] = static_cast<int32_t>(sum);
        }
        c[row] = carry;
    }
}

__global__ void subtract_kernel_32_256(
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> a,
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> b,
    torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> o
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < a.size(0)) {
        uint64_t borrow = 0;
        for (int i = 0; i < 8; ++i) {
            int64_t diff = static_cast<int64_t>(a[idx][i]) - 
                           static_cast<int64_t>(b[idx][i]) - 
                           borrow;
            
            o[idx][i] = static_cast<int32_t>(diff);

            borrow = (diff < 0) ? 1 : 0;
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void compare_tensors_cuda(
    const torch::Tensor& a, 
    const torch::Tensor& b, 
    torch::Tensor& o
) {
    int MAX_THRDS = 512;
    int rows = a.size(0);
    int thrds = std::min(rows, MAX_THRDS);
    int blocks = (rows + thrds - 1) / thrds;
    compare_kernel_32_256<<<blocks, thrds>>>(
        a.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        b.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        o.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>()
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA compare kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    }

}


void add_tensors_cuda(
    const torch::Tensor& a, 
    const torch::Tensor& b, 
    torch::Tensor& o,
    torch::Tensor& c
) {
    int MAX_THRDS = 512;
    int rows = a.size(0);
    int thrds = std::min(rows, MAX_THRDS);
    int blocks = (rows + thrds - 1) / thrds;
    add_kernel_32_256<<<blocks, thrds>>>(
        a.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        b.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        o.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        c.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>()
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA compare kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    }

}

void sub_tensors_cuda(
    const torch::Tensor& a, 
    const torch::Tensor& b, 
    torch::Tensor& o
) {
    int MAX_THRDS = 512;
    int rows = a.size(0);
    int thrds = std::min(rows, MAX_THRDS);
    int blocks = (rows + thrds - 1) / thrds;
    subtract_kernel_32_256<<<blocks, thrds>>>(
        a.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        b.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        o.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>()
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA compare kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    }

}


torch::Tensor modular_add_cuda(
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& m
) {


    auto rows = a.size(0);
    auto options = a.options();
    auto sum = torch::zeros({rows, 8}, options);
    auto carry_out = torch::zeros({rows}, options.dtype(torch::kInt32));
    auto cmp_result = torch::zeros({rows}, options.dtype(torch::kInt32));
    auto result = torch::zeros({rows, 8}, options);

    int threads = 256;
    int blocks = (rows + threads - 1) / threads;

    // Step 1: Perform addition
    add_kernel_32_256<<<blocks, threads>>>(
        a.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        b.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        sum.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        carry_out.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>()
    );

    // Synchronize to ensure addition is complete
    cudaDeviceSynchronize();

    // Step 2: Compare sum with modulus m
    compare_kernel_32_256<<<blocks, threads>>>(
            sum.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            m.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            cmp_result.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>()
        );

    // Synchronize to ensure comparison is complete
    cudaDeviceSynchronize();

    // Step 3: Perform conditional subtraction
    subtract_kernel_32_256<<<blocks, threads>>>(
            sum.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            m.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            result.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>()
        );

    // Synchronize to ensure subtraction is complete
    cudaDeviceSynchronize();

    // Step 4: Choose the correct result based on comparison
    // If sum >= m, use the subtracted result; else, use the sum
    auto final_result = torch::where(
        cmp_result.unsqueeze(1).expand({rows, 8}).to(torch::kBool),
        result,
        sum
    );

    return final_result;
}