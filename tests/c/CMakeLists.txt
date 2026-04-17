add_executable(test
    #linear.c
    #mse.c
    #dataset.c
    #train.c
    omp.c
)

target_link_libraries(test
    PRIVATE
        ml
        OpenMP::OpenMP_C
)

find_package(OpenMP REQUIRED)