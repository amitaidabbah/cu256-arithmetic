#ifndef KERNELS_H
#define KERNELS_H

#include <torch/torch.h>

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

#endif // KERNELS_H
