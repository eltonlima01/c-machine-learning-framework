# add_executable(test linear.c)
add_executable(test omp.c)

find_package(OpenMP REQUIRED)

target_link_libraries(test
    PRIVATE
        ml
        OpenMP::OpenMP_C
)