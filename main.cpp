#include <cuda_runtime.h>
#include <torch/types.h>
#include <iostream>
#include "test.h"

int main()
{
    test_comp_256bit();
    test_add_256bit();
    test_sub_256bit();
    test_modular_add_256bit();
}
