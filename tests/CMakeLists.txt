# add_executable(test linear.c)
add_executable(test omp.c)

target_link_libraries(test
    PRIVATE
        ml
)