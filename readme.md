PyTorch CUDA 256-bit Integer Operations
=======================================

Overview
--------
This project demonstrates the implementation of 256-bit integer operations (addition, subtraction, comparison, and modular addition) using CUDA and PyTorch. The code is written in C++ with CUDA kernels to leverage GPU acceleration for efficient large integer arithmetic. This project showcases how to use CUDA with PyTorch's PackedTensorAccessor32 for efficient GPU tensor operations.

Features
--------
- 256-bit Integer Addition: Supports addition of 256-bit integers with carry handling.
- 256-bit Integer Subtraction: Supports subtraction with borrow handling.
- 256-bit Integer Comparison: Compares two 256-bit integers.
- 256-bit Modular Addition: Adds two 256-bit integers modulo a third 256-bit integer.

Requirements
------------
- CMake (version 3.10 or higher)
- CUDA Toolkit (version 11.0 or higher)
- PyTorch C++ API (LibTorch)
- A CUDA-capable GPU (Compute Capability 7.5 or higher recommended)
- C++17 compiler (e.g., g++, clang)

Installation
------------

Step 1: extract the zip file
Step 2: Install Dependencies
Make sure you have the following dependencies installed:
- CUDA Toolkit: https://developer.nvidia.com/cuda-toolkit
- LibTorch: https://pytorch.org/get-started/locally/

If LibTorch is not installed, download it and extract it to a directory (e.g., `/usr/local/libtorch`).

Step 3: Build the Project
Create a `build` directory and run CMake to generate the build files:
 mkdir build
 cd build
 cmake -DLIBTORCH_PATH=/path/to/libtorch ..
 make

