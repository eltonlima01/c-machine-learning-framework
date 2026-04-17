add_executable(test
    #linear.c
    #logistic.c
    #dataset.c
    #mse.c
    train.c
)

target_link_libraries(test
    ml
)