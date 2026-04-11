# C Machine Learning Framework

A lightweight Machine Learning framework written in pure C from scratch, built with Modern CMake and OpenMP.

## Project Structure

```
c-machine-learning-framework/
├── CMakeLists.txt     # Root build configuration
├── ml/
│   ├── CMakeLists.txt # Core build configuration
│   ├── ml.h           # Single-Header Public API
│   ├── dataset.c      # Dataset loading
│   └── linear.c       # Linear Regression & Gradient Descent logic
└── tests/
    ├── datasets/      # Example datasets
    ├── CMakeLists.txt
    ├── linear.c       # Linear Regression model & prediction tests
    ├── dataset.c      # Dataset loading test
    ├── mse.c          # MSE calculation test
    └── train.c        # Model training test
```

## Dependencies

- [OpenMP](https://www.openmp.org/)
> The OpenMP® API is a scalable model that gives programmers a simple and flexible interface for developing portable parallel applications in C/C++ and Fortran. OpenMP is suitable for a range of algorithms running on multicore nodes and chips, NUMA systems, GPUs, and other such devices attached to a CPU.

- [CMake](https://cmake.org/)
> CMake is the de-facto standard for building C++ code, with over 2 million downloads a month. It’s a powerful, comprehensive solution for managing the software build process.