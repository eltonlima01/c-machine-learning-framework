cmake_minimum_required(VERSION 3.23)

project(C_Machine_Learning_Framework C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED true)

add_subdirectory(ml)
add_subdirectory(tests)