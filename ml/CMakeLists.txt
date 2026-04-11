cmake_minimum_required(VERSION 3.23)

add_library(ml STATIC)

set_target_properties(ml PROPERTIES
    C_STANDARD 99
    C_STANDARD_REQUIRED true
)

find_package(OpenMP REQUIRED)

target_sources(ml
    PRIVATE
        linear.c
        dataset.c

    PUBLIC
        FILE_SET HEADERS
        BASE_DIRS .
        FILES
            ml.h
)

target_link_libraries(ml
    PUBLIC
        OpenMP::OpenMP_C
)