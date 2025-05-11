cmake_minimum_required(VERSION 3.10)
project(NTTMatrix)

if (MSVC)
    add_compile_options(/bigobj)
endif()

add_subdirectory(vendors)

add_subdirectory(tests)
add_subdirectory(examples)
