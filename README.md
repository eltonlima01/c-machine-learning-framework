<div align="center">

# C Machine Learning Framework

![C](https://img.shields.io/badge/language-C99%2B-blue.svg)
![CMake](https://img.shields.io/badge/CMake-3.23%2B-success.svg)
![OpenMP](https://img.shields.io/badge/OpenMP-Enabled-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

</div>

A high-performance, zero-dependency Machine Learning framework implemented from scratch in pure C, utilizing OpenMP for parallel workload distribution and Modern CMake for scalable build management.

## Current Features

- Unified **Model** interface for the initialization, management, and training of simple Linear and Logistic Regressions.
- Built-in **Dataset** constructor with basic CSV data parsing & loading capabilities.
- Low-level computation **kernels** for high-performance math, leveraging parallel processing via native SIMD/OpenMP integration.

## Dependencies

- [CMake](https://cmake.org/) (VERSION 3.23+)
> CMake is the de-facto standard for building C++ code, with over 2 million downloads a month. It’s a powerful, comprehensive solution for managing the software build process.

- A C compiler with [OpenMP](https://www.openmp.org/) support
> The OpenMP® API is a scalable model that gives programmers a simple and flexible interface for developing portable parallel applications in C/C++ and Fortran. OpenMP is suitable for a range of algorithms running on multicore nodes and chips, NUMA systems, GPUs, and other such devices attached to a CPU.

## Project Structure

| Directory | Description |
| :--- | :--- |
| **`/core`** | Main framework library, pure C API & core logic implementation |
| **`/src`** | SIMD/OpenMP mathematical kernels implementation |
| **`/cpp-package`** | C++ Object-Oriented wrapper interface & implementation |
<!-- | **`/tests`** | Unit and integration tests for core logic & API validation | -->