cmake_minimum_required(VERSION 3.23)

add_library(ml STATIC)

set_target_properties(ml PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED True
    LINKER_LANGUAGE CXX
)

target_sources(ml
    PRIVATE
        Model.cpp
        Dataset.cpp

    PUBLIC
        FILE_SET HEADERS
        BASE_DIRS .
        FILES
            ml.hpp
)

target_link_libraries(ml
    ml_kernels
)