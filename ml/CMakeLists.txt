cmake_minimum_required(VERSION 3.23)

add_library(ml STATIC)

set_target_properties(ml PROPERTIES
    C_STANDARD 99
    C_STANDARD_REQUIRED True
)

target_sources(ml
    PRIVATE
        ml.c
        dataset.c

    PUBLIC
        FILE_SET HEADERS
        BASE_DIRS .
        FILES
            ml.h
)

find_package(OpenMP REQUIRED)

target_link_libraries(ml
    PRIVATE
        m   
        OpenMP::OpenMP_C
)