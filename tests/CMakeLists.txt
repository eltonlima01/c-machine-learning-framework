# add_executable(test linear.c)
add_executable(test mse.c)

target_link_libraries(test
    PRIVATE
        ml
)