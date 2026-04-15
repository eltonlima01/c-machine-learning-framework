#add_executable(test linear.c)
#add_executable(test omp.c)
#add_executable(test linear.cpp)
#add_executable(test dataset.cpp)
add_executable(test train.cpp)

target_include_directories(test
    PRIVATE
        cpp-package
)

#find_package(OpenMP REQUIRED)

target_link_libraries(test
    PRIVATE
        ml
        #OpenMP::OpenMP_C
)