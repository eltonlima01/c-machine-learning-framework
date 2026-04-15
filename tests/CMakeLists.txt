#add_executable(test linear.c)
#add_executable(test omp.c)
#add_executable(test linear.cpp)
add_executable(test dataset.cpp)

#find_package(OpenMP REQUIRED)

target_include_directories(test
    PRIVATE
        cpp-package
)

target_link_libraries(test
    PRIVATE
        ml
        #OpenMP::OpenMP_C
)