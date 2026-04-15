cmake_minimum_required(VERSION 3.23)

add_library(ml_kernels STATIC)

set_target_properties(ml_kernels PROPERTIES
    C_STANDARD 99
    C_STANDARD_REQUIRED True
)

target_sources(ml_kernels
    PRIVATE
        math_kernels.c

    PUBLIC
        FILE_SET HEADERS
        BASE_DIRS .
        FILES
            math_kernels.h
)

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ml_kernels
        PRIVATE
            -O3
            -march=native
            -ffast-math
    )
endif()

find_package(OpenMP REQUIRED)

target_link_libraries(ml_kernels
    PRIVATE
        OpenMP::OpenMP_C
)